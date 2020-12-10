include(bazel)

cc_binary(
  NAME "examples"
  HDRS 
    "example1.h"
  SRCS "main.cc"
  DEPS
    "CONAN_PKG::glog"
    "intellgraph"
)
