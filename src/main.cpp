#include <iostream>
#include <math.h>
#include <vector>

#include "node/node_factory.h"
#include "utility/random.h"
//#include "engine/graph_engine.h"

using namespace std;
using namespace intellgraph;

int main() {
  auto node_param1 = NodeParameter(0, "SigmoidNode_f", {2,1});
  auto node_param2 = NodeParameter(1, "SigmoidNode_f", {2,1});
  auto node_param3 = NodeParameter(2, "SigmoidNode_f", {2,1});
  auto node_param4 = NodeParameter(3, "SigL2Node_f", {1,1});

  NodeUPtr<float> sigmoid_ptr = NodeFactory<float, Node<float>>::Instantiate(node_param1);
  sigmoid_ptr->PrintAct();
}
