load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

tf_custom_op_library(
    name = "nccl_ops.so",
    srcs = [
        "nccl_op_kernels.cc",
        "nccl_ops.cc",
        "nccl_optimizer.cc",
    ],
    # TODO(pbar) This won't work in google3!
    # Link aginst NCCL installed in local file system.
    #linkopts = [
    #    "-L/usr/local/lib",
    #    "-lnccl",
    #],
    # Link against prebuilt binary package.
    deps = [
        ":libnccl",
        "//tensorflow/core:core_cpu_headers_lib",
    ],
)

# Prebuilt binaries of NCCL.
# TODO(pbar) Needs to build from source in third_party.
cc_library(
    name = "libnccl",
    srcs = [
        "libnccl.so",
    ],
    hdrs = [
        "nccl.h",
    ],
)
