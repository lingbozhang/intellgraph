#include <iostream>
#include <math.h>
#include <vector>

#include "edge/edge_factory.h"
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
  node1_ptr = NodeFactory<float, NodeSPtr<float>>::Instantiate("SigmoidNode_f", 
                                                               node_param1);
  OutputNodeSPtr<float> node2_ptr;
  node2_ptr = NodeFactory<float, OutputNodeSPtr<float>>::Instantiate(
      "SigL2Node_f", node_param2);

  struct EdgeParameter<float> edge_param1;
  edge_param1.id = 0;
  edge_param1.in_node_ptr = node1_ptr;
  edge_param1.out_node_ptr = node2_ptr;

  EdgeSPtr<float> edge_ptr;
  edge_ptr = EdgeFactory<float>::Instantiate("DenseEdge_f", edge_param1);
  edge_ptr->GetWeightPtr()->array() << -2, -2;
  node2_ptr->GetBiasPtr()->array() << 3;

  node1_ptr->GetActivationPtr()->array() << 0, 1;
  edge_ptr->Forward();
  node2_ptr->PrintAct();
  MatXXSPtr<float> data_result_ptr = make_shared<MatXX<float>>(1,1);

  node2_ptr->CalcLoss(data_result_ptr);

  return 0;
}
