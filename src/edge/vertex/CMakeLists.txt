cc_library(
  STATIC
  NAME "vertex"
  HDRS
    "input_vertex_impl.h"
    "op_vertex_impl.h"
    "output_vertex_impl.h"
    "relu.h"
    "sigmoid.h"
    "sigmoid_l2.h"
  SRCS
    "input_vertex_impl.cc"
    "op_vertex_impl.cc"
    "output_vertex_impl.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "proto"
)

cc_test(
  NAME "vertex_unittests"
  SRCS
    "input_vertex_impl_test.cc"
    "op_vertex_impl_test.cc"
    "output_vertex_impl_test.cc"
    "relu_test.cc"
    "sigmoid_test.cc"
    "sigmoid_l2_test.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "vertex"
)

# Installs IntellGraph include headers
install(
  FILES 
    input_vertex_impl.h
    op_vertex_impl.h
    output_vertex_impl.h
    relu.h
    sigmoid.h
    sigmoid_l2.h
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
)
