include(bazel)

cc_library(
  STATIC
  NAME "solver"
  HDRS
    "sgd_solver.h"
  SRCS
    "sgd_solver.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
    "edge"
)

# Installs IntellGraph include headers
install(
  FILES sgd_solver.h 
  DESTINATION ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
) 
