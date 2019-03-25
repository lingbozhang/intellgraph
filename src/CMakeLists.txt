cmake_minimum_required(VERSION 3.2)
project(IntellGraph)

add_subdirectory(node)
add_subdirectory(edge)
add_subdirectory(utility)

set(SOURCE_FILES main.cpp)

add_executable(IntellGraph ${SOURCE_FILES})
target_link_libraries(IntellGraph node edge utility)

install(TARGETS IntellGraph DESTINATION ${INTELLGRAPH_INSTALL_BIN_DIR})
