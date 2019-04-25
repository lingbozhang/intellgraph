/* Copyright 2019 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
	Huicheng Zhang <huichengz0520@gmail.com>
==============================================================================*/
#ifndef INTELLGRAPH_GRAPH_RNN_GRAPH_H_
#define INTELLGRAPH_GRAPH_RNN_GRAPH_H_

#include "node/node_headers.h"
#include "edge/edge_headers.h"

namespace intellgraph {
//
// (I)-0-->
//        |
//        --> (2) ---3-->(O)
//        |   |
// (1)-1-->   |
//  |         |
//  <----2----
//
template <class T>
class RNNGraph {
 public:
  RNNGraph() {}
  ~RNNGraph() {}
  
//  void Forward(REF const Eigen::Ref<const MatXX<T>>& training_data, \
//      REF const Eigen::Ref<const MatXX<T>>& training_labels) {
//    input_node_.FeedFeature(training_data);
//    //node_2_.get_activation_ptr()->matrix() =
//    edge_0_.Forward(&input_node_, &node_2_);
//  }
  // BackPropagate();
  // Evaluate();
  // get_output_ptr;
 private:
  Node<T> input_node_{};
  Node<T> output_node_{};
  Node<T> node_1_{};
  Node<T> node_2_{};
  Edge<T> edge_0_{};
  Edge<T> edge_1_{};
  Edge<T> edge_2_{};
  Edge<T> edge_3_{};
  size_t T{0};
};

}  // intellgraph

#endif  // INTELLGRAPH_GRAPH_RNN_GRAPH_H_