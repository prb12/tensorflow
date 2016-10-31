#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/op.h"

//REGISTER_RESOURCE_HANDLE_OP(NcclCommResource);

REGISTER_OP("NcclCommResourceHandleOp")
      .Attr("container: string = ''")
      .Attr("shared_name: string = ''")
      .Output("comm: resource")
      .SetIsStateful()
  //      .SetShapeFn([](shape_inference::InferenceContext* c) {   c->set_output(0, c->Scalar());  return Status::OK();})
      .Doc("Creates a handle to a NcclCommResource.");

REGISTER_OP("NcclInitComm")
    .Input("comm: resource")
    .Attr("rank: int")
    .Attr("N: int")
    .Attr("id: int");
//    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("NcclReduce")
    .Input("comm: resource")
    .Input("input: float32")
    .Output("sum: float32");

REGISTER_OP("NcclAllReduce")
    .Input("comm: resource")
    .Input("input: float32")
    .Output("sum: float32");

REGISTER_OP("NcclBroadcast")
    .Input("comm: resource")
    .Input("input: float32")
    .Output("sum: float32");
