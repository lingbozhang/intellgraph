cmake_minimum_required(VERSION 3.10)

project(graph)

file(GLOB graph_source *.cc)
file(GLOB graph_header *.h)

add_library(graph SHARED STATIC ${graph_source})

install(TARGETS graph DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${graph_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})

