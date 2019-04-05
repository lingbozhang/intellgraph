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
	Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#ifndef INTELLGRAPH_EXAMPLE_SIMPLE_CLASSIFIER_H_
#define INTELLGRAPH_EXAMPLE_SIMPLE_CLASSIFIER_H_

#include <iostream>
#include <memory>

#include "edge/edge_headers.h"
#include "graph/classifier.h"
#include "graph/graph.h"
#include "node/node_headers.h"
#include "utility/common.h"

using namespace std;
using namespace intellgraph;
// Example 1 constructs and trains a simple classifer. The classifier has two
// layers: input layer and output layer. In the output layer, the sigmoid 
// function is used as the activation function and l2 norm is used as the loss
// function. 
class Example1 {
 public:
  static void run() {
    // Prepares train data
    auto train_d_ptr = std::make_shared<MatXX<float>>(2, 2);
    auto train_l_ptr = std::make_shared<MatXX<float>>(1, 2);

    // In IntellGraph, current version implements Eigen library for matrix
    // abstraction.
    train_d_ptr->array() << 0.0, 1.0,
                            0.0, 1.0;

    train_l_ptr->array() << 0.0, 1.0;

    NodeRegistry::LoadNodeRegistry();
    EdgeRegistry::LoadEdgeRegistry();

    // SigInputNode is an input layer which uses Sigmoid function as activation
    // function. Note Node has two dimensions, the first dimension indicates
    // number of nodes and the second dimension is currently used for batch
    // size
    auto node_param1 = NodeParameter<float>(0, "SigInputNode", {2, 1});

    // SigL2Node uses Sigmoid function as activation function and l2 norm as
    // loss function.
    auto node_param2 = NodeParameter<float>(1, "SigL2Node", {1, 1});

    // IntellGraph implements Boost Graph library and stores node and edge
    // information in the adjacency list.
    Classifier<float> classifier;
    // DenseEdge represents fully connected networks
    classifier.AddEdge(node_param1, node_param2, "DenseEdge");

    classifier.set_input_node_id(0);
    classifier.set_output_node_id(1);

    classifier.Instantiate();

    MatXXSPtr<float> train_data_ptr = std::make_shared<MatXX<float>>(1, 1);
    MatXXSPtr<float> train_label_ptr = std::make_shared<MatXX<float>>(1, 1);

    float eta = 1;
    float lambda = 0;
    int loops = 1500;
    int data_num = 2;
    for (int epoch = 0; epoch < loops; ++epoch) {
      std::cout << "Epoch: " << epoch << "/" << loops << std::endl;
      int i = rand() % data_num;
      train_data_ptr->array() = train_d_ptr->col(i);
      train_label_ptr->array() = train_l_ptr->col(i);
      classifier.Backward(train_data_ptr, train_label_ptr);
      // Stochastic gradient decent
      classifier.get_edge_weight(0, 1)->array() = \
        (1.0 - eta * lambda) * classifier.get_edge_weight(0, 1)->array() - \
        eta * classifier.get_edge_nabla(0, 1)->array();
      classifier.get_node_bias(1)->array() -= eta * \
        classifier.get_node_delta(1)->array();
    }
    std::cout << "Complete" << std::endl;
  }
};

#endif  // INTELLGRAPH_EXAMPLE_SIMPLE_CLASSIFIER_H_
 