include(bazel)

cc_library(
  STATIC
  NAME "intellgraph"
  HDRS
    "classifier_impl.h"
    "graph_builder.h"
  SRCS
    "classifier_impl.cc"
    "graph_builder.cc"
  PUBLIC_DEPS
    "CONAN_PKG::eigen"
    "edge"
    "factory"
    "proto"
    "solver"
    "vertex"
    "utility"
    "visitor"
  DEPS
    "CONAN_PKG::boost"
    "CONAN_PKG::glog"
)

install(
  TARGETS intellgraph 
  DESTINATION ${INTELLGRAPH_LIB_DIR}/intellgraph
)

# Installs IntellGraph include headers
install(
  FILES 
    classifier_impl.h 
    graph_builder.h 
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph/graph
)
