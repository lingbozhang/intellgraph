add_subdirectory(edge)
add_subdirectory(graph)
add_subdirectory(proto)
add_subdirectory(visitor)
add_subdirectory(solver)
add_subdirectory(utility)

cc_library(
  STATIC
  NAME "factory"
  HDRS
    "factory.h"
    "registry.h"
  SRCS
    "registry.cc"
  DEPS
    "CONAN_PKG::glog"
    "edge"
    "proto"
    "solver"
    "vertex"
)

# Installs IntellGraph include headers
install(
  FILES 
    edge.h
    eigen.h
    graph.h 
    registry.h
    visitor.h
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
) 
