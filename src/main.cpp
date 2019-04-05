#include <iostream>
#include <math.h>
#include <vector>

#include "edge/edge_headers.h"
#include "graph/classifier.h"
#include "graph/graph.h"
#include "node/node_headers.h"
#include "utility/random.h"
#include "utility/common.h"

using namespace std;
using namespace intellgraph;

int main() {
  // A simple two-layer deep learning example
  // Prepares train data
  auto train_d_ptr = std::make_shared<MatXX<float>>(2,4);
  auto train_l_ptr = std::make_shared<MatXX<float>>(1,4);

  train_d_ptr->array() << 0.0, 1.0, 0.0, 1.0,
                          0.0, 0.0, 1.0, 1.0;

  train_l_ptr->array() << 1, 1, 1, 0;

  NodeRegistry::LoadNodeRegistry();
  EdgeRegistry::LoadEdgeRegistry();

  // Prepares classifiers
  auto node_param1 = NodeParameter<float>(0, "SigInputNode", {2, 1});
  auto node_param2 = NodeParameter<float>(1, "SigL2Node", {1, 1});

  Classifier<float> classifier;
  classifier.AddEdge(node_param1, node_param2, "DenseEdge");

  classifier.set_input_node_id(0);
  classifier.set_output_node_id(1);
  // Instantiates classifier
  classifier.Instantiate();

  MatXXSPtr<float> train_data_ptr = std::make_shared<MatXX<float>>(2, 1);
  MatXXSPtr<float> train_label_ptr = std::make_shared<MatXX<float>>(1, 1);

  float eta = 0.5;
  int loops = 2000;
  for (int epoch = 0; epoch < loops; ++epoch) {
    std::cout << "Epoch: " << epoch << "/" << loops << std::endl;
    size_t i = rand() % 4;
    train_data_ptr->array() = train_d_ptr->col(i);
    train_label_ptr->array() = train_l_ptr->col(i);
    classifier.Backward(train_data_ptr, train_label_ptr);
    // Stochastic Learning
    classifier.get_node_bias(1)->array() -= eta * classifier.get_node_delta(1)->array();
    classifier.get_edge_weight(0, 1)->array() -= eta * classifier.get_edge_nabla(0, 1)->array();
  }
  std::cout << "Complete" << std::endl;
}
