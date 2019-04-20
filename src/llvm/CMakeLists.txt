cmake_minimum_required(VERSION 3.10)
project(llvm)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs Support ExecutionEngine Interpreter Core nativecodegen MC MCJIT)

add_executable(llvm main.cc)
# Link against LLVM libraries
target_link_libraries(llvm ${llvm_libs})

# Pass MNIST data directory to main.cpp
#target_compile_definitions(examples PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

