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

#include <cstdio>

#include <iostream>
#include <memory>

#include "include/intellgraph/eigen.h"
#include "include/intellgraph/graph/classifier_impl.h"
#include "include/intellgraph/graph/graph_builder.h"
#include "include/intellgraph/proto/vertex_parameter.pb.h"
#include "include/intellgraph/registry.h"
#include "include/intellgraph/solver/sgd_solver.h"
#include "src/solver/adagrad.h"
#include <google/protobuf/text_format.h>

namespace intellgraph {

// Example 1 constructs a two-layer (including the Input Layer) neural network
// that predicts the gender based on the weight and height. The IntellGraph has
// two vertices: input vertex and output vertex. In the output vertex, the cross
// entropy is used as the loss function.
class Example1 {
public:
  Example1() = delete;
  ~Example1() = delete;

  static void Run() {
    std::cout << "====================================================\n"
              << "  Predict gender based on body weight and height    \n"
              << "====================================================\n"
              << "Name    Weight(minus 135) Height(minus 66)  Gender  \n"
              << "Alice   -2                -1                1       \n"
              << "Bob     25                6                 0       \n"
              << "Charlie 17                4                 0       \n"
              << "Diana  -15                -6                1       \n"
              << "Tom     20                5                 0       \n"
              << "====================================================\n";

    // Prepares training data
    MatrixX<float> training_feature(2, 4);
    MatrixX<int> training_labels(1, 4);
    MatrixX<float> test_feature(2, 1);
    MatrixX<int> test_labels(1, 1);
    training_feature << -2, 25, 17, -15, -1, 6, 4, -6;
    training_labels << 1, 0, 0, 1;
    test_feature << 20, 5;
    test_labels << 0;

    // Registering instances
    Registry::LoadRegistry();

    // Constructs vertices
    VertexParameter vtx_param1, vtx_param2;
    // Vertex 1
    google::protobuf::TextFormat::ParseFromString(
        "id: 0 type: INPUT operation: 'DummyTransformer' dims: 2", &vtx_param1);
    // Vertex 2
    google::protobuf::TextFormat::ParseFromString(
        "id: 1 type: OUTPUT operation: 'CrossEntropy' dims: 1", &vtx_param2);

    // Constructs a solver
    SolverConfig solver_config;
    google::protobuf::TextFormat::ParseFromString(
        "type: 'SGD' eta: 0.01 lambda: 0.0", &solver_config);

    // Builds the graph
    // The Dense edge represents a fully connected neural network
    GraphBuilder<float> graph_builder;
    ClassifierImpl<float> classifier =
        graph_builder
            .AddEdge(/*edge_id=*/0, /*edge_type=*/"Dense", vtx_param1,
                     vtx_param2)
            .AddSolver(solver_config)
            .SetLength(1)
            .BuildClassifier();

    classifier.SetSolver(std::make_unique<Adagrad<float>>(0.01, 0.0));
    int epochs = 500;
    std::cout << "Total epochs: " << epochs << std::endl;
    int total_size = training_feature.cols();
    for (int epoch = 0; epoch < epochs; ++epoch) {
      for (int i = 0; i < 4; ++i) {
        classifier.Train(training_feature.col(i), training_labels.col(i));
      }
      float loss = classifier.CalculateLoss(training_feature, training_labels);
      printf("Epoch %4d/%4d, Loss: %e\n", epoch, epochs, loss);
    }
    std::cout << "Training complete!!!" << std::endl;
    float gender =
        classifier.GetProbabilityDist(test_feature).array().round()(0, 0);
    printf("Tom's predicted gender: %1.0f (0: male, 1: female)\n", gender);
  }
};

} // namespace intellgraph

#endif // INTELLGRAPH_EXAMPLES_EXAMPLE1_H_
