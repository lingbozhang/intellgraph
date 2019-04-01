cmake_minimum_required(VERSION 3.10)

project(engine)

#file(GLOB node_source *.cpp)
file(GLOB engine_header *.h)

add_library(engine SHARED STATIC graph_engine.cc)


install(TARGETS engine DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${engine_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})