include(bazel)

cc_library(
  STATIC
  NAME "solver"
  HDRS
    "momentum.h"
    "sgd_solver.h"
  SRCS
    "momentum.cc"
    "sgd_solver.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "edge"
    "proto"
)

# Installs IntellGraph include headers
install(
  FILES 
    momentum.h
    sgd_solver.h 
  DESTINATION ${INTELLGRAPH_INCLUDE_DIR}/intellgraph/solver
) 
