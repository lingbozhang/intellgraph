cmake_minimum_required(VERSION 3.10)

project(solver)

#file(GLOB solver_source *.cc)
file(GLOB solver_header *.h)

#add_library(solver SHARED STATIC ${solver_source})

#install(TARGETS solver DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${solver_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR}/solver)