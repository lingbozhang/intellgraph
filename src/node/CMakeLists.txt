cmake_minimum_required(VERSION 3.10)

project(node)

#file(GLOB node_source *.cpp)
file(GLOB node_header *.h)

add_library(node SHARED STATIC sigmoid_node.cc activation_node.cc sigmoid_l2_node.cc act_loss_node.cc sigmoid_input_node.cc)

install(TARGETS node DESTINATION ${INTELLGRAPH_INSTALL_LIB_DIR})
install(FILES ${node_header} DESTINATION ${INTELLGRAPH_INSTALL_INCLUDE_DIR})