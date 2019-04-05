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
#include "engine/graph_engine.h"

namespace intellgraph {

template <class T>
void GraphEngine<T>::AddEdge(const NodeParameter<T>& node_param_in, \
                             const NodeParameter<T>& node_param_out, \
                             const std::string& edge_name) {
  // Construct node objects and put them in the node_map_
  size_t vertex_in_id = node_param_in.ref_id();
  size_t vertex_out_id = node_param_out.ref_id();

  EdgeParameter edge_param{};
  EdgeProperty edge_property{};

  if (node_param_map_.count(vertex_in_id) == 0) {
    node_param_map_[vertex_in_id].Clone(node_param_in);
  }

  if (node_param_map_.count(vertex_out_id) == 0) {
    node_param_map_[vertex_out_id].Clone(node_param_out);
  }
  edge_property.id = edge_param_map_.size();

  // Constructs EdgeParameter
  edge_param.set_id(edge_property.id);
  edge_param.set_edge_name(edge_name);
  edge_param.set_dims_in(node_param_in.ref_dims());
  edge_param.set_dims_out(node_param_out.ref_dims());

  if (!boost::add_edge(vertex_in_id, vertex_out_id, edge_property, graph_). \
      second) {
    std::cout << "WARNING: Edge has already been added" << std::endl;
    return;
  }
  edge_param_map_[edge_property.id].Clone(edge_param);
}

template <class T>
void GraphEngine<T>::Instantiate() {
  node_map_.clear();
  edge_map_.clear();
  input_node_ptr_ = nullptr;
  output_node_ptr_ = nullptr;

  // Instantiates the input node object
  auto input_node_ptr = std::move(NodeFactory<T, InputNode<T>>::Instantiate( \
      node_param_map_[input_node_id_]));
  input_node_ptr_ = input_node_ptr.get();
  node_map_[input_node_id_] = std::move(input_node_ptr);
  
  // Instantiates the output node object
  auto output_node_ptr = std::move(NodeFactory<T, OutputNode<T>>::Instantiate( \
      node_param_map_[output_node_id_]));
  output_node_ptr_ = output_node_ptr.get();
  node_map_[output_node_id_] = std::move(output_node_ptr);

  output_node_ptr_->InitializeBias(NormalFunctor<T>(0.0, 1.0));

  // Instantiates node objects;
  for (const auto& node_pair : node_param_map_) {
    size_t node_id = node_pair.first;
    if (node_id != output_node_id_ && node_id != input_node_id_) {
      NodeUPtr<T> node_ptr = \
          std::move(NodeFactory<T, Node<T>>::Instantiate(node_pair.second));
      node_ptr->InitializeBias(NormalFunctor<T>(0.0, 1.0));
      node_map_[node_pair.first] = std::move(node_ptr);
    }
  }
  // Instantiates edge objects;
  for (const auto& edge_pair : edge_param_map_) {
    EdgeUPtr<T> edge_ptr = std::move(EdgeFactory<T, Edge<T>>::Instantiate( \
        edge_pair.second));
    // Initializes weight matrix with standard normal distribution
    edge_ptr->InitializeWeight(NormalFunctor<T>(0.0, 1.0));
    edge_map_[edge_pair.first] = std::move(edge_ptr);
  }
}

template <class T>
void GraphEngine<T>::Forward(MUTE MatXXSPtr<T> train_data_ptr) {
  order_.clear();
  topological_sort(graph_, std::back_inserter(order_));

  if (*order_.rbegin() != input_node_id_ || \
    *order_.begin() != output_node_id_) {
      std::cout << "ERROR: invalid graph." << std::endl;
      exit(1);
  }

  input_node_ptr_->FeedFeature(train_data_ptr);
  for (auto it_r = order_.rbegin(); it_r != order_.rend(); ++it_r) {
    IntellGraph::out_edge_iterator eo{}, eo_end{};
    size_t node_in_id = *it_r;
    //std::cout << "Forwarding node: " << node_in_id << std::endl;
    if (*it_r != input_node_id_ ) node_map_[*it_r]->CallActFxn();
    for (std::tie(eo, eo_end) = out_edges(*it_r, graph_); eo != eo_end; ++eo) {
      size_t node_out_id = target(*eo, graph_);
      size_t edge_id = graph_[*eo].id;

      Node<T> *node_in_ptr, *node_out_ptr;

      node_in_ptr = node_map_[node_in_id].get();
        
      node_out_ptr = node_map_[node_out_id].get();
 
      edge_map_[edge_id]->Forward(node_in_ptr, node_out_ptr);
    }
  }
}

template <class T>
void GraphEngine<T>::Backward(MUTE MatXXSPtr<T> train_label_ptr, \
                              float learning_rate) {
  output_node_ptr_->CalcDelta(train_label_ptr.get());
  for (auto it = order_.begin(); it != order_.end(); ++it) {
    IntellGraph::in_edge_iterator ei{}, ei_end{};
    size_t node_out_id = *it;
    //std::cout << "Backpropagating node: " << node_out_id << std::endl;
    for (std::tie(ei, ei_end) = in_edges(*it, graph_); ei != ei_end; ++ei) {
      size_t node_in_id = source(*ei, graph_);
      size_t edge_id = graph_[*ei].id;

      Node<T> *node_in_ptr, *node_out_ptr;
      Edge<T> *edge_ptr;

      node_in_ptr = node_map_[node_in_id].get();       
      node_out_ptr = node_map_[node_out_id].get();
      edge_ptr = edge_map_[edge_id].get();

      edge_ptr->Backward(node_in_ptr, node_out_ptr);
      node_out_ptr->get_bias_ptr()->array() -= learning_rate * \
          node_out_ptr->get_delta_ptr()->array();
      edge_ptr->get_weight_ptr()->array() -= learning_rate * \
          edge_ptr->get_nabla_weight_ptr()->array();
    }
  }
}

template <class T>
void GraphEngine<T>::Learn(MUTE MatXXSPtr<T> train_data_ptr, \
                           MUTE MatXXSPtr<T> train_label_ptr, \
                           float learning_rate) {
  Forward(train_data_ptr);
  T loss = output_node_ptr_->CalcLoss(train_label_ptr.get());
  std::cout << "Square Loss: " << loss << std::endl;
  Backward(train_label_ptr, learning_rate);
}

// Instantiate class, otherwise compilation will fail
template class GraphEngine<float>;
template class GraphEngine<double>;
}  // intellgraph