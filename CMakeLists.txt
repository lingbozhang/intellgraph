cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(INTELLGRAPH_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(INTELLGRAPH_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(INTELLGRAPH_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(INTELLGRAPH_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories(${INTELLGRAPH_INSTALL_INCLUDE_DIR})
include_directories(${INTELLGRAPH_HEADERS_DIR})
include_directories(${INTELLGRAPH_EXAMPLES_DIR})

include_directories(third_party/boost)
include_directories(third_party/eigen-git-mirror)
include_directories(third_party/glog/src)
include_directories(third_party/mnist/include)

set(MNIST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/mnist/include)
set(MNIST_DATA_DIR ${PROJECT_SOURCE_DIR}/third_party/mnist)
set(MNIST_FOUND TRUE)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(third_party/glog)


