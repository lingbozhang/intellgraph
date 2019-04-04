cmake_minimum_required(VERSION 3.10)

project(engine)

file(GLOB engine_source *.cc)
file(GLOB engine_header *.h)

add_library(engine SHARED STATIC ${engine_source})


install(TARGETS engine DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${engine_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})