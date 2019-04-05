cmake_minimum_required(VERSION 3.10)
project(IntellGraph)

add_subdirectory(graph)
add_subdirectory(edge)
add_subdirectory(node)
add_subdirectory(utility)

set(SOURCE_FILES main.cpp)

add_executable(IntellGraph ${SOURCE_FILES})
target_link_libraries(IntellGraph graph node edge glog::glog)

install(TARGETS IntellGraph DESTINATION ${INTELLGRAPH_INSTALL_BIN_DIR})
