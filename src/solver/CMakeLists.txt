include(bazel)

cc_library(
  STATIC
  NAME "solver"
  HDRS
    "adadelta.h"
    "adagrad.h"
    "adam.h"
    "ada_max.h"
    "momentum.h"
    "sgd_solver.h"
  SRCS
    "adadelta.cc"
    "adagrad.cc"
    "adam.cc"
    "ada_max.cc"
    "momentum.cc"
    "sgd_solver.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "edge"
    "proto"
    "utility"
)

# Installs IntellGraph include headers
install(
  FILES 
    adadelta.h
    adagrad.h
    adam.h
    ada_max.h
    momentum.h
    sgd_solver.h 
  DESTINATION ${INTELLGRAPH_INCLUDE_DIR}/intellgraph/solver
) 
