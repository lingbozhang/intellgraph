#/* Copyright 2019 The IntellGraph Authors. All Rights Reserved.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#
#Contributor(s):
#	Lingbo Zhang <lingboz2015@gmail.com>
#==============================================================================*/

# This file contains cmake functions that support bazel-like syntax
cmake_minimum_required(VERSION 3.13)

# cc_binary for build an executable target:
# cc_binary(
#  <NAME>
#  <SRCS>...
#  [HDRS]...
#  [DEPS|PUBLIC_DEPS]...
#)
function(cc_binary)
  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS HDRS DEPS PUBLIC_DEPS)
  cmake_parse_arguments(
    CC_BINARY
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  if(NOT CC_BINARY_NAME)
    message(FATAL_ERROR "Binary name is not specified")
  endif()
  if(NOT CC_BINARY_SRCS)
    message(FATAL_ERROR "Source files are not specified for target: 
        ${CC_BINARY_NAME}")
  endif()
  add_executable(${CC_BINARY_NAME} ${CC_BINARY_SRCS} ${CC_BINARY_HDRS})
  if(CC_BINARY_DEPS) 
    target_link_libraries(${CC_BINARY_NAME} PRIVATE ${CC_BINARY_DEPS})
  endif()
  if(CC_BINARY_PUBLIC_DEPS)
    target_link_libraries(${CC_BINARY_NAME} PUBLIC ${CC_BINARY_PUBLIC_DEPS})
  endif()
  set_target_properties(${CC_BINARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# cc_library for build a library:
# cc_library(
#  [STATIC|SHARED]
#  <NAME>
#  <SRCS>...
#  [HDRS]...
#  [DEPS|PUBLIC_DEPS]...
#)
function(cc_library)
  set(options STATIC SHARED)
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS HDRS DEPS PUBLIC_DEPS)
  cmake_parse_arguments(
    CC_LIBRARY
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  if(NOT CC_LIBRARY_NAME)
    message(FATAL_ERROR "Library name is not specified")
  endif()
  if(NOT CC_LIBRARY_SRCS)
    message(FATAL_ERROR "Source files are not specified for target: 
        ${CC_LIBRARY_NAME}")
  endif()
  if(CC_LIBRARY_STATIC)
    add_library(${CC_LIBRARY_NAME} STATIC ${CC_LIBRARY_SRCS} ${CC_LIBRARY_HDRS})
  elseif(CC_LIBRARY_SHARED)
    add_library(${CC_LIBRARY_NAME} SHARED ${CC_LIBRARY_SRCS} ${CC_LIBRARY_HDRS})
  else()
    add_library(${CC_LIBRARY_NAME} ${CC_LIBRARY_SRCS} ${CC_LIBRARY_HDRS})
  endif()
  if(CC_LIBRARY_DEPS) 
    target_link_libraries(${CC_LIBRARY_NAME} PRIVATE ${CC_LIBRARY_DEPS})
  endif()
  if(CC_LIBRARY_PUBLIC_DEPS)
    target_link_libraries(${CC_LIBRARY_NAME} PUBLIC ${CC_LIBRARY_PUBLIC_DEPS})
  endif()
  set_target_properties(${CC_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# proto_library for generate and build protobuf C++ library:
# proto_library(
#  [STATIC|SHARED]
#  <NAME>
#  <SRCS>...
#  [DEPS|PUBLIC_DEPS]...
#)
function(proto_library)
  set(options STATIC SHARED)
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS DEPS PUBLIC_DEPS)
  cmake_parse_arguments(
    PROTO_LIBRARY
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  if(NOT PROTO_LIBRARY_NAME)
    message(FATAL_ERROR "Library name is not specified")
  endif()
  if(NOT PROTO_LIBRARY_SRCS)
    message(FATAL_ERROR "Source files are not specified for target: 
    ${PROTO_LIBRARY_NAME}")
  endif()
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_LIBRARY_SRCS})
  if(PROTO_LIBRARY_STATIC)
    add_library(${PROTO_LIBRARY_NAME} STATIC ${PROTO_SRCS} ${PROTO_HDRS})
  elseif(PROTO_LIBRARY_SHARED)
    add_library(${PROTO_LIBRARY_NAME} SHARED ${PROTO_SRCS} ${PROTO_HDRS})
  else()
    add_library(${PROTO_LIBRARY_NAME} ${PROTO_SRCS} ${PROTO_HDRS})
  endif()
  if(PROTO_LIBRARY_DEPS) 
    target_link_libraries(${PROTO_LIBRARY_NAME} PRIVATE ${PROTO_LIBRARY_DEPS} ${PROTOBUF_LIBRARIES})
  endif()
  if(PROTO_LIBRARY_PUBLIC_DEPS)
    target_link_libraries(${PROTO_LIBRARY_NAME} PUBLIC ${PROTO_LIBRARY_PUBLIC_DEPS} ${PROTOBUF_LIBRARIES})
  endif()
  target_link_libraries(${PROTO_LIBRARY_NAME} PRIVATE ${PROTOBUF_LIBRARIES})
  set_target_properties(${PROTO_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# cc_test for build a test
function(cc_test)
  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(
    CC_TEST
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  if(NOT CC_TEST_NAME)
    message(FATAL_ERROR "Test name is not specified")
  endif()
  if(NOT CC_TEST_SRCS)
    message(FATAL_ERROR "Source files are not specified for target: 
        ${CC_TEST_NAME}")
  endif()
  add_executable(${CC_TEST_NAME} ${CC_TEST_SRCS})
  target_link_libraries(${CC_TEST_NAME} PRIVATE gtest gmock gtest_main
      ${CC_TEST_DEPS})
  target_include_directories(${CC_TEST_NAME} PRIVATE
      ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
  )
  gtest_discover_tests(${CC_TEST_NAME})
endfunction()
