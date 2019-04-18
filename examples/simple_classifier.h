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
#include "graph/graphfxn.h"
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
    std::cout << "=================================" << std::endl;
    std::cout << "A Simple Classifier for 2D points" << std::endl;
    std::cout << "=================================" << std::endl;
    // Prepares train data
    MatXX<float> training_data(2, 6);
    MatXX<float> training_labels(1, 6);
    MatXX<float> test_data(2, 4);
    MatXX<float> test_labels(1, 4);

    // In IntellGraph, current version implements Eigen library for matrix
    // abstraction.
    // Training data: * indicates 1, O indicates 0.0;
    //
    //                 (0.5, 1.0)(*)   (1.0, 1.0)(*)
    //  (0.0, 0.5)(O)                  (1.0, 0.5)(*)
    //  (0.0, 0.0)(O)  (0.5, 0.0)(O)
    //
    training_data << 0.0, 1.0, 0.5, 1.0, 0.0, 0.5,
                     0.0, 1.0, 1.0, 0.5, 0.5, 0.0;

    training_labels << 0.0, 1.0, 1, 1, 0.0, 0.0;

    test_data << 0.25,  0.0, 0.25, 0.75,
                  0.0, 0.25, 0.25, 0.75;

    test_labels << 0.0, 0.0, 0.0, 1.0;


    NodeRegistry::LoadNodeRegistry();
    EdgeRegistry::LoadEdgeRegistry();

    // SigmoidNode is an internal node which uses Sigmoid function as activation
    // function. 
    //auto node_param1 = NodeParameter(0, "SigmoidNode", {2});

    // SigL2Node uses Sigmoid function as activation function and l2 norm as
    // loss function.
    //auto node_param2 = NodeParameter(1, "SigL2Node", {1});

    // IntellGraph implements Boost Graph library and stores node and edge
    // information in the adjacency list.
    Classifier<float> classifier;
    classifier.set_edge({"SigmoidNode", 2}, {"SigL2Node", 1}, "DenseEdge");
    // DenseEdge represents fully connected networks
    //classifier.AddEdge(node_param1, node_param2, "DenseEdge");

    //classifier.set_input_node_id(0);
    //classifier.set_output_node_id(1);

    classifier.Instantiate();

    float eta = 1;
    int loops = 50;
    std::cout << "Learning rate: " << eta << std::endl;
    std::cout << "Total epochs: " << loops << std::endl;
    int data_num = 6;
    for (int epoch = 0; epoch < loops; ++epoch) {
      std::cout << "Epoch: " << epoch << "/" << loops << std::endl;
      for (int i = 0; i < 6; ++i) {
        int col = rand() % data_num;
        classifier.Derivative(training_data.col(col), training_labels.col(col));
        // Stochastic gradient decent
        classifier.get_edge_weight_ptr(0, 1)->array() -= \
            eta * classifier.get_edge_nabla_ptr(0, 1)->array();
        classifier.get_node_bias_ptr(1)->array() -= eta * \
            classifier.get_node_delta_ptr(1)->array();
      }
      classifier.Evaluate(test_data, test_labels);
    }
    std::cout << "Complete" << std::endl;
  }
};

#endif  // INTELLGRAPH_EXAMPLE_SIMPLE_CLASSIFIER_H_
 