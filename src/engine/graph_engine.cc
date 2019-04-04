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
  size_t vertex_in_id = node_param_in.get_k_id();
  size_t vertex_out_id = node_param_out.get_k_id();

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
  edge_param.set_c_id(edge_property.id);
  edge_param.set_c_edge_name(edge_name);
  edge_param.set_c_dims_in(node_param_in.get_k_dims());
  edge_param.set_c_dims_out(node_param_out.get_k_dims());

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
  // Instantiates the outputnode object
  OutputNodeUPtr<T> output_node_ptr_ = std::move( \
      NodeFactory<T, OutputNode<T>>::Instantiate( \
          node_param_map_[output_node_id_]));
  // Instantiates the inputnode object
  InputNodeUPtr<T> input_node_ptr_ = std::move( \
      NodeFactory<T, InputNode<T>>::Instantiate( \
          node_param_map_[input_node_id_]));

  // Instantiates node objects;
  for (auto& node_pair : node_param_map_) {
    size_t node_id = node_pair.first;
    if (node_id != output_node_id_ && node_id != input_node_id_) {
      NodeUPtr<T> node_ptr = 
          std::move(NodeFactory<T, Node<T>>::Instantiate(node_pair.second));
      node_map_[node_pair.first] = std::move(node_ptr);
    }
  }
  // Instantiates edge objects;
  for (auto& edge_pair : edge_param_map_) {
    EdgeUPtr<T> edge_ptr = std::move(EdgeFactory<T, Edge<T>>::Instantiate( \
        edge_pair.second));
    // Initializes weight matrix with standard normal distribution
    edge_ptr->ApplyUnaryFunctor_k(NormalFunctor<T>(0.0, 1.0));
    edge_map_[edge_pair.first] = std::move(edge_ptr);
  }
}

// Instantiate class, otherwise compilation will fail
template class GraphEngine<float>;
template class GraphEngine<double>;
}  // intellgraph