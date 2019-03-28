#include <iostream>
#include <math.h>
#include <vector>

#include "engine/graph_engine.h"
#include "utility/common.h"
#include "utility/random.h"

using namespace std;
using namespace intellgraph;

int main() {
  struct NodeParameter node_param1, node_param2;
  node_param1.id = 0;
  node_param1.node_name = "SigmoidNode_f";
  node_param1.dims.push_back(2);

  node_param2.id = 1;
  node_param2.node_name = "SigmoidNode_f";
  node_param2.dims.push_back(1);

  GraphEngine<float> graph;
  //NodeSPtr<float> node_in_ptr = NodeFactory<float, NodeSPtr<float>>::Instantiate(node_param1);
  graph.AddEdge(node_param1, node_param2, "DenseEdge_f");
  graph.node_param_map_[0];

  return 0;
}
