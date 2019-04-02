cmake_minimum_required(VERSION 3.10)

project(utility)

#file(GLOB utility_source *.cc)
file(GLOB utility_header *.h)

#add_library(utility SHARED STATIC ${utility_source})

#install(TARGETS utility DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${utility_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})