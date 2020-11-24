include(bazel)

proto_library(
  STATIC
  NAME "proto"
  SRCS 
    "vertex_parameter.proto"
    "edge_parameter.proto"
)

# Installs IntellGraph include headers
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DESTINATION ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
)
