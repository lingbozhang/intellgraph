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
#include <functional>
#include <iostream>

#include "glog/logging.h"
#include "src/edge/registry.h"
#include "src/graph/graph_builder.h"
#include "src/proto/vertex_parameter.pb.h"

using namespace intellgraph;

// In IntellGraph, current version implements Eigen library for data
// abstraction.
int main(int argc, char *argv[]) {
  // Initializes Google's logging library.
  FLAGS_alsologtostderr = false;
  FLAGS_minloglevel = 1;
  fLS::FLAGS_log_dir = "/tmp/";
  google::InitGoogleLogging(argv[0]);

  // Prepares train data
  MatrixX<float> training_data(2, 6);
  MatrixX<float> training_labels(1, 6);
  MatrixX<float> test_data(2, 4);
  MatrixX<float> test_labels(1, 4);

  // Training data: * represents class 1, O represents class 0;
  //
  //                 * (0.5, 1.0)   * (1.0, 1.0)
  //  O (0.0, 0.5)                  * (1.0, 0.5)
  //  O (0.0, 0.0)   O (0.5, 0.0)
  //
  training_data << 0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0;

  training_labels << 0, 1, 1, 1, 0, 0;

  test_data << 0.25, 0.0, 0.25, 0.75, 0.0, 0.25, 0.25, 0.75;

  test_labels << 0.0, 0.0, 0.0, 1.0;

  // Registering instances
  Registry::LoadRegistry();

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
  GraphBuilder<float> graph_builder;
  std::unique_ptr<GraphImpl<float>> graph =
      graph_builder.AddEdge("Dense", vtx_param1, vtx_param2)
          .SetInputVertexId(0)
          .SetOutputVertexId(1)
          .SetBatchSize(1)
          .Build();

  float eta = 1.0;
  SgdSolver<float> solver(1, 0.0);
  int loops = 50;
  std::cout << "=================================" << std::endl;
  std::cout << "A Simple Classifier for 2D points" << std::endl;
  std::cout << "=================================" << std::endl;
  std::cout << "Learning rate: " << eta << std::endl;
  std::cout << "Total epochs: " << loops << std::endl;
  std::cout << "=================================" << std::endl;
  int training_num = 6;
  for (int epoch = 0; epoch < loops; ++epoch) {
    std::cout << "Epoch: " << epoch << "/" << loops << std::endl;
    for (int i = 0; i < 6; ++i) {
      int col = rand() % training_num;
      MatrixX<float> data = training_data.col(col);
      MatrixX<float> label = training_labels.col(col);
      graph->Train(solver, data, label);
    }
    std::cout << "Accuracy: "
              << graph->CalculateAccuracy(test_data, test_labels) << " %"
              << std::endl;
  }
  std::cout << "Complete" << std::endl;
  return 0;
}
