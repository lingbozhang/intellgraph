cmake_minimum_required(VERSION 3.10)
project(examples)

LINK_DIRECTORIES(${INTELLGRAPH_INSTALL_LIB_DIR}/node)
LINK_DIRECTORIES(${INTELLGRAPH_INSTALL_LIB_DIR}/edge)
LINK_DIRECTORIES(${INTELLGRAPH_INSTALL_LIB_DIR}/graph)

set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/mnist-master)

set(SOURCE_FILES main.cc)

add_executable(examples ${SOURCE_FILES})
target_link_libraries(examples node edge graph ${CONAN_LIBS})

# Pass MNIST data directory to main.cpp
target_compile_definitions(examples PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

