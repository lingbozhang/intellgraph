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

  auto node_param1 = NodeParameter<float>(0, "SigInputNode", {2,4});
  auto node_param2 = NodeParameter<float>(0, "SigL2Node", {1,4});

  auto edge_param1 = EdgeParameter(0, "DenseEdge", {2,1}, {1,1});

  auto node1_ptr = NodeFactory<float, InputNode<float>>::Instantiate( \
      node_param1);
  auto node2_ptr = NodeFactory<float, OutputNode<float>>::Instantiate( \
      node_param2);

  auto edge1_ptr = EdgeFactory<float, Edge<float>>::Instantiate(edge_param1);
  edge1_ptr->get_c_weight_ptr()->array() << -2, -2;
  node2_ptr->get_c_bias_ptr()->array() << 3, 3, 3, 3;

  auto train_d_ptr = std::make_shared<MatXX<float>>(2,4);
  auto train_l_ptr = std::make_shared<MatXX<float>>(1,4);

  train_d_ptr->array() << 0.0, 1.0, 0.0, 1.0,
                          0.0, 0.0, 1.0, 1.0;

  train_l_ptr->array() << 1, 1, 1, 0;

  node1_ptr->FeedFeature_k(train_d_ptr);

  edge1_ptr->Forward_mute(node1_ptr.get(), node2_ptr.get());

  node2_ptr->PrintAct();

  node2_ptr->CallActFxn();
  //std::cout << node2_ptr->CalcLoss_k(*train_l_ptr) << std::endl;
  std::cout << std::exp(-1)/ (1.0 + std::exp(-1)) << std::endl;
  node2_ptr->PrintAct();







  //graph_engine.node_map_[1]->get_c_bias_ptr()->array() << 3;


  




  //graph_engine.Forward();
  //std::vector<size_t> order = graph_engine.get_k_typological_order();
  //
  //std::cout << "Typological order" << std::endl;
  //for (auto ele : order) {
  //  std::cout << ele << std::endl;
  //}
}
