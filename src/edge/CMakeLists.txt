cmake_minimum_required(VERSION 3.10)

project(edge)

file(GLOB edge_source *.cc)
file(GLOB edge_header *.h)

add_library(edge SHARED STATIC ${edge_source})

install(TARGETS edge DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR}/edge)
install(FILES ${edge_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR}/edge)