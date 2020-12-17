cc_library(
  STATIC
  NAME "vertex"
  HDRS
    "input_vertex.h"
    "input_vertex_impl.h"
    "op_vertex_impl.h"
    "output_vertex_impl.h"
    "seq_output_impl.h"
    "seq_vertex_impl.h"
  SRCS
    "input_vertex.cc"
    "input_vertex_impl.cc"
    "op_vertex_impl.cc"
    "output_vertex_impl.cc"
    "seq_output_impl.cc"
    "seq_vertex_impl.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "proto"
    "tensor"
)

cc_test(
  NAME "vertex_unittests"
  SRCS
    "input_vertex_impl_test.cc"
    "op_vertex_impl_test.cc"
    "output_vertex_impl_test.cc"
    "relu_test.cc"
    "sigmoid_l2_test.cc"
    "sigmoid_test.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "vertex"
)

# Installs IntellGraph include headers
install(
  FILES 
    cross_entropy.h
    input_vertex_impl.h
    op_vertex_impl.h
    output_vertex_impl.h
    relu.h
    seq_output_impl.h
    seq_vertex_impl.h
    sigmoid.h
    sigmoid_l2.h
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph/edge/vertex
)
