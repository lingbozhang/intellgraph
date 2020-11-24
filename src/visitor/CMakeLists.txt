include(bazel)

cc_library(
  STATIC
  NAME "visitor"
  HDRS
    "backward_visitor.h"
    "forward_visitor.h"
    "init_vertex_visitor.h"
    "normal_init_visitor.h"
    "resize_vertex_visitor.h"
  SRCS
    "backward_visitor.cc"
    "forward_visitor.cc"
    "init_vertex_visitor.cc"
    "normal_init_visitor.cc"
    "resize_vertex_visitor.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "edge"
    "utility"
)

cc_test(
  NAME "visitor_unittests"
  SRCS
    "backward_visitor_test.cc"
    "forward_visitor_test.cc"
    "init_vertex_visitor_test.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "visitor"
)
