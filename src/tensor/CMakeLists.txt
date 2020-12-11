cc_library(
  STATIC
  NAME "tensor"
  HDRS
    "dyn_matrix.h"
  SRCS
    "dyn_matrix.cc"
  DEPS
    "CONAN_PKG::eigen"
    "CONAN_PKG::glog"
)

# Installs IntellGraph include headers
install(
  FILES 
    dyn_matrix.h
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph/tensor
) 
