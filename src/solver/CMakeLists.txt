include(bazel)

cc_library(
  STATIC
  NAME "solver"
  HDRS
    "adagrad.h"
    "momentum.h"
    "sgd_solver.h"
  SRCS
    "adagrad.cc"
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
    adagrad.h
    momentum.h
    sgd_solver.h 
  DESTINATION ${INTELLGRAPH_INCLUDE_DIR}/intellgraph/solver
) 
