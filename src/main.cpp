#include <iostream>
#include <math.h>
#include <vector>

#include "edge/dense_edge.h"
#include "edge/edge_factory.h"
#include "edge/edge_parameter.h"
#include "edge/edge_registry.h"
#include "edge/edge.h"
#include "engine/graph_engine.h"
#include "node/input_node.h"
#include "node/node_edge_interface.h"
#include "node/node_factory.h"
#include "node/node_parameter.h"
#include "node/node_registry.h"
#include "node/node.h"
#include "node/output_node.h"
#include "node/sigmoid_input_node.h"
#include "node/sigmoid_l2_node.h"
#include "node/sigmoid_node.h"
#include "utility/random.h"
#include "utility/common.h"

using namespace std;
using namespace intellgraph;

int main() {
  auto train_d_ptr = std::make_shared<MatXX<float>>(2,4);
  auto train_l_ptr = std::make_shared<MatXX<float>>(1,4);

  train_d_ptr->array() << 0.0, 1.0, 0.0, 1.0,
                          0.0, 0.0, 1.0, 1.0;

  train_l_ptr->array() << 1, 1, 1, 0;

  GraphEngine<float> graph;
  NodeRegistry::LoadNodeRegistry();
  EdgeRegistry::LoadEdgeRegistry();

  auto node_param1 = NodeParameter<float>(0, "SigInputNode", {2, 1});
  auto node_param2 = NodeParameter<float>(1, "SigmoidNode", {3, 1});
  auto node_param3 = NodeParameter<float>(2, "SigL2Node", {1, 1});

  graph.AddEdge(node_param1, node_param2, "DenseEdge");
  graph.AddEdge(node_param2, node_param3, "DenseEdge");

  graph.set_input_node_id(0);
  graph.set_output_node_id(2);

  graph.Instantiate();

  auto data_ptr = std::make_shared<MatXX<float>>(2,1);
  data_ptr->array() << 2.0,
                       1.0;

  graph.output_node_ptr_->PrintAct();
  graph.Forward(data_ptr);
  graph.output_node_ptr_->PrintAct();
}
