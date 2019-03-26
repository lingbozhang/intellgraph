#include <iostream>
#include <math.h>
#include <vector>

#include "edge/output_edge.h"
#include "node/node_factory.h"
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
  node1_ptr = NodeFactory<float>::Instantiate("SigmoidNode_f", node_param1);
  NodeSPtr<float> node2_ptr;
  node2_ptr = NodeFactory<float>::Instantiate("SigL2Node_f", node_param2);

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

>>>>>>> d76c2bb621cca5a81180e952f3c648001287819d

  return 0;
}
