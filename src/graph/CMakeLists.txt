include(bazel)

cc_library(
  STATIC
  NAME "intellgraph"
  HDRS
    "graph_builder.h"
    "graph_impl.h"
  SRCS
    "graph_builder.cc"
    "graph_impl.cc"
  PUBLIC_DEPS
    "CONAN_PKG::eigen"
    "edge"
    "edge_factory"
    "proto"
    "solver"
    "vertex"
    "utility"
  DEPS
    "CONAN_PKG::boost"
    "CONAN_PKG::glog"
    "visitor"
)

install(
  TARGETS intellgraph 
  DESTINATION ${INTELLGRAPH_LIB_DIR}/intellgraph
)

# Installs IntellGraph include headers
install(
  FILES 
    graph_impl.h 
    graph_builder.h 
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
)
