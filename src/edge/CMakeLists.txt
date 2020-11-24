include(bazel)

add_subdirectory(vertex)

cc_library(
  STATIC
  NAME "edge"
  HDRS
    "dense_edge_impl.h"
  SRCS
    "dense_edge_impl.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "vertex"
)

cc_library(
  STATIC
  NAME "edge_factory"
  HDRS
    "factory.h"
    "registry.h"
  SRCS
    "registry.cc"
  DEPS
    "CONAN_PKG::glog"
    "edge"
    "proto"
    "vertex"
)

# Installs IntellGraph include headers
install(
  FILES
    dense_edge_impl.h
    registry.h
  DESTINATION
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
)
