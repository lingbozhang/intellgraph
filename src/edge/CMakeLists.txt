cmake_minimum_required(VERSION 3.12)

project(edge)

file(GLOB edge_source *.cpp)
file(GLOB edge_header *.h)

add_library(edge SHARED STATIC dense_edge.cc output_edge.cc)


install(TARGETS edge DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${edge_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})