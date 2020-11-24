include(bazel)

cc_binary(
  NAME "examples"
  HDRS 
    "example1.h"
    "example2.h"
  SRCS "main.cc"
  DEPS
    "CONAN_PKG::glog"
    "intellgraph"
)

add_custom_command(
  TARGET examples 
  POST_BUILD
  COMMAND examples
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Runs examples target after it is built"
  VERBATIM
)
