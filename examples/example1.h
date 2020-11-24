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
#ifndef INTELLGRAPH_EXAMPLES_EXAMPLE1_H_
#define INTELLGRAPH_EXAMPLES_EXAMPLE1_H_

#include <iostream>
#include <memory>

#include "include/intellgraph/eigen.h"
#include "include/intellgraph/graph_builder.h"
#include "include/intellgraph/graph_impl.h"
#include "include/intellgraph/proto/vertex_parameter.pb.h"
#include "include/intellgraph/registry.h"
#include "include/intellgraph/sgd_solver.h"

namespace intellgraph {

// Example 1 constructs and trains a simple classifer. The classifier has two
// layers: input layer and output layer. In the output layer, the sigmoid
// function is used as the activation function and l2 norm is used as the loss
// function.
// In IntellGraph, current version implements Eigen library for matrix
// abstraction.
class Example1 {
public:
  static void run() {
    std::cout << "=================================" << std::endl;
    std::cout << "A Simple Classifier for 2D points" << std::endl;
    std::cout << "=================================" << std::endl;

    // Prepares train data
    MatrixX<float> training_feature(2, 6);
    MatrixX<float> training_labels(1, 6);
    MatrixX<float> test_feature(2, 4);
    MatrixX<float> test_labels(1, 4);

    // Training data: * represents class 1, O represents class 0;
    //
    //                 * (0.5, 1.0)   * (1.0, 1.0)
    //  O (0.0, 0.5)                  * (1.0, 0.5)
    //  O (0.0, 0.0)   O (0.5, 0.0)
    //
    training_feature << 0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.5,
        0.0;
    training_labels << 0, 1, 1, 1, 0, 0;

    test_feature << 0.25, 0.0, 0.25, 0.75, 0.0, 0.25, 0.25, 0.75;
    test_labels << 0.0, 0.0, 0.0, 1.0;

    // Registering instances
    Registry::LoadRegistry();

    // Constructs vertices
    VertexParameter vtx_param1, vtx_param2;
    // Vertex 1
    vtx_param1.set_id(0);
    vtx_param1.set_type("DummyTransformer");
    vtx_param1.set_dims(2);
    // Vertex 2
    vtx_param2.set_id(1);
    vtx_param2.set_type("SigmoidL2");
    vtx_param2.set_dims(1);

    // Builds the graph
    // The Dense edge is added into the graph which represents a fully connected
    // neural network
    GraphBuilder<float> graph_builder;
    std::unique_ptr<GraphImpl<float>> graph =
        graph_builder.AddEdge("Dense", vtx_param1, vtx_param2)
            .SetInputVertexId(0)
            .SetOutputVertexId(1)
            .SetBatchSize(1)
            .Build();

    // Constructs a solver using the Stochastic Gradient Descent algorithm
    float eta = 1.0;
    SgdSolver<float> solver(eta, 0.01 /* lambda */);

    int loops = 100;
    std::cout << "Total epochs: " << loops << std::endl;
    int total_size = training_feature.cols();
    for (int epoch = 0; epoch < loops; ++epoch) {
      std::cout << "Epoch: " << epoch << "/" << loops << std::endl;
      for (int i = 0; i < 6; ++i) {
        graph->Train(solver, training_feature.col(i), training_labels.col(i));
      }
      float accuracy = graph->CalculateAccuracy(test_feature, test_labels);
      std::cout << "Accuracy: " << accuracy << "%." << std::endl;
      std::cout << "Loss: " << graph->CalculateLoss(test_feature, test_labels)
                << std::endl;
    }
    std::cout << "Complete" << std::endl;
  }
};

} // namespace intellgraph

#endif // INTELLGRAPH_EXAMPLES_EXAMPLE1_H_
