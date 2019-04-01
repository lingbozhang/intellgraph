cmake_minimum_required(VERSION 3.10)
project(IntellGraph)

#add_subdirectory(engine)
#add_subdirectory(edge)
add_subdirectory(node)
add_subdirectory(utility)

set(SOURCE_FILES main.cpp)

add_executable(IntellGraph ${SOURCE_FILES})
target_link_libraries(IntellGraph node utility)

install(TARGETS IntellGraph DESTINATION ${INTELLGRAPH_INSTALL_BIN_DIR})
