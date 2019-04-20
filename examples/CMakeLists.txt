cmake_minimum_required(VERSION 3.10)
project(examples)

set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/mnist-master)

set(SOURCE_FILES main.cc)

add_executable(examples ${SOURCE_FILES})
target_link_libraries(examples graph node edge ${CONAN_LIBS})

# Pass MNIST data directory to main.cpp
target_compile_definitions(examples PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

