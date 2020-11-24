cmake_minimum_required(VERSION 3.13)

project(intellgraph2 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Exports compilation database for VIM-LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sets Intellgraph installation directories
set(INTELLGRAPH_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(INTELLGRAPH_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(INTELLGRAPH_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

# Conan package manager setup
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)
# Finds Protobuf
include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
set(Protobuf_IMPORT_DIRS ${Protobuf_IMPORT_DIRS} ${PROJECT_SOURCE_DIR})
if(DEFINED Protobuf_IMPORT_DIRS)
  foreach(DIR ${PROTOBUF_IMPORT_DIRS})
    get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
    list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
    if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
    endif()
  endforeach()
endif()
# boost
include_directories(${CONAN_INCLUDE_DIRS_BOOST})
# eigen 
include_directories(${CONAN_INCLUDE_DIRS_EIGEN})
# glog
include_directories(${CONAN_INCLUDE_DIRS_GLOG})
# gtest
add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)
message(STATUS ${gtest_SOURCE_DIR})
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR})
# Generated codes such as protobuf are accessible globally
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(src)
add_subdirectory(examples)
