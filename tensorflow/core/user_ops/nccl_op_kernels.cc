#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/cuda.h"

#include "nccl.h"

using namespace tensorflow;

namespace {

#if false
// TODO(pbar) It may be necessary to use ncclCommInitAll to avoid
// creating all the shared memory segments used by multi-process nccl comm.
class NcclCommunicator : public ResourceBase {
 public:
  NcclCommunicator(const std::vector<int>& devices)
  {
    VLOG(1) << "NcclCommunicator create";
    comms_ = new ncclComm_t[devices.size()];
    devices_ = devices;
    VLOG(1) << "ncclCommInitAll ...";
    ncclResult_t ret = ncclCommInitAll(comms_, devices.size(), devices_.data());
    VLOG(1) << "ncclCommInitAll done";
    CHECK_EQ(ret, ncclSuccess);
  }

  ncclComm_t comm(int i) {
    return comms_[i];
}

  string DebugString() { return "NcclCommunicator"; }
  ~NcclCommunicator() = default;
 private:
  int num_devices_;
  ncclComm_t* comms_;
  std::vector<int> devices_;
};
#endif

static ncclUniqueId global_id;

class NcclCommResource : public ResourceBase {
 public:
  NcclCommResource(int rank, int num, int id)
      : rank_(rank), num_devices_(num), comm_id_(id), comm_(NULL)
  {
    VLOG(1) << "NcclCommResource create " << rank << "/" << num;
    // TODO(pbar) Work out how to deal with the ID.
  }

  Status Init() {
    ncclResult_t ret;
    static bool init_done = init_once();
    CHECK(init_done);
    VLOG(1) << "Calling InitRank " << rank_ << " / " << num_devices_;
    VLOG(1) << "Before comm_ = " << comm_;
    ret = ncclCommInitRank(&comm_, num_devices_, global_id, rank_);
    VLOG(1) << "After comm_ = " << comm_;
    CHECK_EQ(ret, ncclSuccess);
    cudaStreamCreate(&stream_);
    cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    return Status::OK();
  }

  ncclComm_t comm() {
    return comm_;
  }

  cudaStream_t stream() {
    return stream_;
  }

  cudaEvent_t event() {
    return event_;
  }
  string DebugString() { return "NcclComm"; }
  ~NcclCommResource() {
    cudaEventDestroy(event_);
    VLOG(1) << "NcclCommResource destructor.";
  }
 private:
  bool init_once() {
    // TODO(pbar) This is the ID shared by all tasks in a
    // communicator clique.  Since we can't share resources across TF
    // devices currently, we are using a single global id.
    // This means there can only be one clique of fixed size.
    VLOG(1) << "Init UniqueId";
    ncclResult_t ret = ncclGetUniqueId(&global_id);
    CHECK_EQ(ret, ncclSuccess);
    VLOG(1) << "Got UniqueId";
    return true;
  }
  int rank_;
  int num_devices_;
  int comm_id_;
  ncclComm_t comm_;
  cudaStream_t stream_;
  cudaEvent_t event_;
};

} // namespace;

// TODO(pbar) Make this use resource handles.
class NcclInitCommOp : public AsyncOpKernel {
 public:
  explicit NcclInitCommOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    VLOG(1) << "NcclInitCommOp constructor";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_devices_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    VLOG(1) << "NcclInitCommOp::Compute " << rank_ << "/" << num_devices_ <<":" << unique_id_;
    auto handle = HandleFromInput(ctx, 0);
    VLOG(1) << "Got handle";
    VLOG(1) << "Handle: " << handle.DebugString();

    auto* comm = new NcclCommResource(rank_, num_devices_, unique_id_);
    OP_REQUIRES_OK(ctx, CreateResource<NcclCommResource>(
        ctx, HandleFromInput(ctx, 0), comm));
    VLOG(1) << "Created resource: comm=" << comm;

    // TOOD(pbar) This is potentially blocking.
    auto* runner = ctx->runner();
    //    gpu::Stream* stream = ctx->op_device_context<GPUDeviceContext>()->stream();
    gpu::Stream* stream = ctx->op_device_context()->stream();
    (*runner)(std::bind(&NcclInitCommOp::InitAsync, this, stream, comm, done));
    VLOG(1) << "Enqueued init";
  }
 private:
  void InitAsync(gpu::Stream* stream, NcclCommResource *comm, DoneCallback done) {
    VLOG(1) << "InitAsync comm=" << comm;
    // NOTE: Async GPU ops don't get run in a proper CUDA context!
    gpu::cuda::ScopedActivateExecutorContext scoped_activation{
      stream->parent()};
    comm->Init();
    VLOG(1) << "Done!";
    done();
  }
  int rank_;
  int num_devices_;
  int unique_id_;
};


class NcclReduceOp : public OpKernel {
 public:
  explicit NcclReduceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "NcclReduceOp::Compute";

    NcclCommResource* comm;
    OP_REQUIRES_OK(ctx, LookupResource<NcclCommResource>(
        ctx, HandleFromInput(ctx, 0), &comm));

    // Grab the input tensor
    const Tensor& input_tensor = ctx->input(1);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
                                             &output_tensor));

    //const void* src_ptr = DMAHelper::base(&input_tensor);
    //void* dst_ptr = DMAHelper::base(output_tensor);
    const void * src_ptr = input_tensor.tensor_data().data();
    const void * dst_ptr = output_tensor->tensor_data().data();
    (void)src_ptr;
    (void)dst_ptr;
  }
};

class NcclUniqueIdOp : public OpKernel {
 public:
  explicit NcclUniqueIdOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "NcclUniqueuId::Compute";
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    string *buffer = &output->scalar<string>()();
    buffer->resize(NCCL_UNIQUE_ID_BYTES);
    ncclUniqueId* unique_id = reinterpret_cast<ncclUniqueId*>(&(*buffer)[0]);
    ncclResult_t ret = ncclGetUniqueId(unique_id);
    CHECK_EQ(ret, ncclSuccess);
    VLOG(1) << "Got UniqueId";
  }
};

class NcclAllReduceOp : public OpKernel {
 public:
  explicit NcclAllReduceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "NcclAllReduceOp::Compute";

    NcclCommResource* comm;
    OP_REQUIRES_OK(ctx, LookupResource<NcclCommResource>(
        ctx, HandleFromInput(ctx, 0), &comm));
    VLOG(1) << "comm: " << comm;

    // Grab the input tensor
    const Tensor& input_tensor = ctx->input(1);
    VLOG(1) << "input shape: " << input_tensor.shape().DebugString();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
                                             &output_tensor));
    const void * src_ptr = input_tensor.tensor_data().data();
    const void * dst_ptr = output_tensor->tensor_data().data();
    const auto count = input_tensor.NumElements();
    VLOG(1) << "Count: " << count;
    VLOG(1) << "src: " << src_ptr;
    VLOG(1) << "dst: " << dst_ptr;
    // TODO(pbar) Can't do this with StreamExecutor yet!
    cudaStream_t stream = ctx->eigen_gpu_device().stream();
    VLOG(1) << "comm->comm(): " << comm->comm();
    VLOG(1) << "stream: " << stream;

    // TODO(pbar) For now, I run all of the nccl ops on distinct streams
    // since they need to execute concurrently and I only have one GPU!.
    // This is BUGGY since it doesn't wait for the inputs to be comuted on
    // the real stream (doing so would deadlock!).
    // In these tests, the inputs are const in GPU memory,
    // so this isn't a problem.
    // DO NOT SUBMIT
    auto ret = ncclAllReduce((void*)src_ptr, (void*)dst_ptr, count,
                             ncclFloat, ncclSum, comm->comm(), comm->stream());
    // Make the TF stream wait on the NCCL stream.
    cudaEventRecord(comm->event(), comm->stream());
    cudaStreamWaitEvent(stream, comm->event(), 0);
    VLOG(1) << "AllReduce returned " << ret;
    CHECK_EQ(ret, ncclSuccess);
  }
};

class NcclBroadcastOp : public OpKernel {
 public:
  explicit NcclBroadcastOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "NcclBroadcastOp::Compute";
    ResourceMgr* rm = ctx->resource_manager();
    OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));

    NcclCommResource* comm;
    OP_REQUIRES_OK(ctx, LookupResource<NcclCommResource>(
        ctx, HandleFromInput(ctx, 0), &comm));

    // Grab the input tensor
    const Tensor& input_tensor = ctx->input(1);
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    const void * src_ptr = input_tensor.tensor_data().data();
    const void * dst_ptr = output_tensor->tensor_data().data();
    auto stream = ctx->eigen_gpu_device().stream();
    const auto count = input_tensor.NumElements();
    (void) src_ptr;  // TODO(pbar) Handle root device differently!
    auto ret = ncclBcast((void*)dst_ptr, count,
                         ncclFloat, 0, comm->comm(), stream);
    CHECK_EQ(ret, ncclSuccess);
  }
 private:
  ncclComm_t comm_;
  int rank_;
  int num_devices_;
};

// Don't register the CPU version...
REGISTER_RESOURCE_HANDLE_KERNEL(NcclCommResource);

// Register a GPU handle opkernel with HostMemory attribute.
REGISTER_KERNEL_BUILDER(Name("NcclCommResourceHandleOp")
                        .Device(DEVICE_GPU)
                        .HostMemory("comm"),
                        ResourceHandleOp<NcclCommResource>);

REGISTER_KERNEL_BUILDER(Name("NcclUniqueId")
                        .Device(DEVICE_CPU), NcclUniqueIdOp);
REGISTER_KERNEL_BUILDER(Name("NcclUniqueId")
                        .Device(DEVICE_GPU)
                        .HostMemory("id"), NcclUniqueIdOp);

REGISTER_KERNEL_BUILDER(Name("NcclInitComm")
                        .Device(DEVICE_GPU)
                        .HostMemory("comm")
                        .HostMemory("id"), NcclInitCommOp);
REGISTER_KERNEL_BUILDER(Name("NcclInitComm")
                        .Device(DEVICE_CPU), NcclInitCommOp);

REGISTER_KERNEL_BUILDER(Name("NcclReduce")
                        .Device(DEVICE_GPU)
                        .HostMemory("comm"), NcclReduceOp);

REGISTER_KERNEL_BUILDER(Name("NcclAllReduce")
                        .Device(DEVICE_GPU)
                        .HostMemory("comm"), NcclAllReduceOp);

REGISTER_KERNEL_BUILDER(Name("NcclBroadcast")
                        .Device(DEVICE_GPU)
                        .HostMemory("comm"), NcclBroadcastOp);
