#include <iostream>
#include <math.h>
#include <vector>

//#include "edge/edge_factory.h"
//#include "engine/graph_engine.h"
#include "node/sigmoid_node.h"
#include "node/node.h"
#include "node/output_node.h"
#include "utility/random.h"
#include "utility/common.h"
#include "utility/registry.h"
//#include "engine/graph_engine.h"

using namespace std;
using namespace intellgraph;



int main() {
  // Create data
  MatXX<float> train_data(4,3);
  MatXX<float> train_label(4,1);
  
  train_data << 0, 0, 1,
                0, 1, 1,
                1, 0, 1,
                1, 1, 1;
  
  train_label << 0,
                 1,
                 1,
                 0;

  Registry::LoadRegistry();
  auto node_param1 = NodeParameter<float>(0, "SigmoidNode_f", {2,1});
  auto node_param2 = NodeParameter<float>(1, "ActivationNode", {1,1});
  auto node_param3 = NodeParameter<float>(2, "SigL2Node", {1,1});
  //SigmoidNode<float> hl;
  NodeUPtr<float> node_ptr = NodeFactory<float, Node<float>>::Instantiate(node_param2);
  node_ptr->CalcActPrime();

  //GraphEngine<float> graph_engine;
  //graph_engine.AddEdge(node_param1, node_param2, "DenseEdge_f");
  //graph_engine.AddEdge(node_param2, node_param3, "DenseEdge_f");
  //graph_engine.set_c_output_node_id(1);

  //graph_engine.Instantiate();
  MatXX<float> train_d(4,2);
  MatXX<float> train_l(4,1);

  train_d << 0, 0,
             1, 0,
             0, 1,
             1, 1;

  train_l << 3,
             1,
             1,
             -1;
  
  //graph_engine.node_map_[1]->get_c_bias_ptr()->array() << 3;


  




  //graph_engine.Forward();
  //std::vector<size_t> order = graph_engine.get_k_typological_order();
  //
  //std::cout << "Typological order" << std::endl;
  //for (auto ele : order) {
  //  std::cout << ele << std::endl;
  //}
}
