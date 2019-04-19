cmake_minimum_required(VERSION 3.10)

project(node)

file(GLOB node_source *.cc)
file(GLOB node_header *.h)

add_library(node SHARED STATIC ${node_source})

install(TARGETS node DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR}/node)
install(FILES ${node_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR}/node)