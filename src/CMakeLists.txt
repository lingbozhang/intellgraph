cmake_minimum_required(VERSION 3.10)
project(IntellGraph)

add_subdirectory(transformer)
add_subdirectory(utility)
add_subdirectory(node)
add_subdirectory(edge)
add_subdirectory(graph)
add_subdirectory(solver)

set(SOURCE_FILES main.cpp)

add_executable(IntellGraph ${SOURCE_FILES})
target_link_libraries(IntellGraph graph node edge ${CONAN_LIBS})

install(TARGETS IntellGraph DESTINATION ${INTELLGRAPH_INSTALL_BIN_DIR})
