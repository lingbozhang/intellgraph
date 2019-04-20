cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(INTELLGRAPH_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(INTELLGRAPH_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(INTELLGRAPH_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
add_subdirectory(examples)
#add_subdirectory(test)


