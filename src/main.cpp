#include <iostream>
#include <math.h>
#include <vector>

#include "edge/edge_parameter.h"
#include "edge/output_edge.h"
#include "node/act_loss_node.h"
#include "node/activation_node.h"
#include "node/node_parameter.h"
#include "node/sigmoid_node.h"
#include "node/sigmoid_l2_node.h"
#include "utility/common.h"
#include "utility/random.h"

using namespace std;
using namespace intellgraph;

int main() {
  struct NodeParameter node_param1, node_param2;
  node_param1.id = 0;
  node_param1.dims.push_back(2);

  node_param2.id = 1;
  node_param2.dims.push_back(1);

  NodeSPtr<float> node1_ptr;
  node1_ptr = make_shared<SigmoidNode<float>>(node_param1);
  OutputNodeSPtr<float> node2_ptr;
  node2_ptr = make_shared<SigL2Node<float>>(node_param2);

  struct EdgeParameter<float> edge_param1;
  edge_param1.id = 0;
  edge_param1.in_node_ptr = node1_ptr;
  edge_param1.out_node_ptr = node2_ptr;

  OutputEdgeSPtr<float> edge_ptr;
  edge_ptr = make_shared<OutputEdge<float>>(edge_param1);
  edge_ptr->GetWeightPtr()->array() << -2, -2;
  node2_ptr->GetBiasPtr()->array() << 3;

  node1_ptr->GetActivationPtr()->array() << 0, 1;
  edge_ptr->Forward();
  node2_ptr->PrintAct();
  
  




  return 0;
}
