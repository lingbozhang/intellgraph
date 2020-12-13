include(bazel)

cc_library(
  STATIC
  NAME "solver"
  HDRS
    "adadelta.h"
    "adagrad.h"
    "momentum.h"
    "sgd_solver.h"
  SRCS
    "adadelta.cc"
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
    adadelta.h
    adagrad.h
    momentum.h
    sgd_solver.h 
  DESTINATION ${INTELLGRAPH_INCLUDE_DIR}/intellgraph/solver
) 
