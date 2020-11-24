# Install script for directory: /Users/lingbozhang/src/intellgraph2/src/edge/vertex

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/lingbozhang/src/intellgraph2/include/intellgraph/input_vertex_impl.h;/Users/lingbozhang/src/intellgraph2/include/intellgraph/op_vertex_impl.h;/Users/lingbozhang/src/intellgraph2/include/intellgraph/output_vertex_impl.h;/Users/lingbozhang/src/intellgraph2/include/intellgraph/relu.h;/Users/lingbozhang/src/intellgraph2/include/intellgraph/sigmoid.h;/Users/lingbozhang/src/intellgraph2/include/intellgraph/sigmoid_l2.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/Users/lingbozhang/src/intellgraph2/include/intellgraph" TYPE FILE FILES
    "/Users/lingbozhang/src/intellgraph2/src/edge/vertex/input_vertex_impl.h"
    "/Users/lingbozhang/src/intellgraph2/src/edge/vertex/op_vertex_impl.h"
    "/Users/lingbozhang/src/intellgraph2/src/edge/vertex/output_vertex_impl.h"
    "/Users/lingbozhang/src/intellgraph2/src/edge/vertex/relu.h"
    "/Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid.h"
    "/Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid_l2.h"
    )
endif()

