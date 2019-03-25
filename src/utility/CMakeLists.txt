cmake_minimum_required(VERSION 3.12)

project(utility)

#file(GLOB utility_source *.cpp)
file(GLOB utility_header *.h)

add_library(utility SHARED STATIC nl_random.cc Common.cc)

install(TARGETS utility DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${utility_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})