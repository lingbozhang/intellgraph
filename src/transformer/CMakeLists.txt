cmake_minimum_required(VERSION 3.10)

project(transformer)

#file(GLOB utility_source *.cc)
file(GLOB transformer_header *.h)

#add_library(utility SHARED STATIC ${utility_source})

#install(TARGETS utility DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${transformer_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})