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
void GraphEngine<T>::AddEdge(const NodeParameter& node_param_in, \
                             const NodeParameter& node_param_out, \
                             const std::string& edge_name) {
  // Construct node objects and put them in the node_map_
  size_t vertex_in_id = node_param_in.id;
  size_t vertex_out_id = node_param_out.id;

  EdgeParameter<T> edge_param;
  EdgeProperty edge_property;

  if (node_param_map_.count(vertex_in_id) == 0) {
    node_param_map_[vertex_in_id] = node_param_in;
  }

  if (node_param_map_.count(vertex_out_id) == 0) {
    node_param_map_[vertex_out_id] = node_param_out;
  }
  edge_property.id = edge_param_map_.size();

  if (edge_param_map_.count(edge_property.id) > 0) {
    std::cout << "ERROR: Edge " << edge_property.id << "has been added"
              << std::endl;
    exit(1);
  } else {
    edge_param.id = edge_property.id;
    edge_param.edge_name = edge_name;
    edge_param.dims_in = node_param_in.dims;
    edge_param.dims_out = node_param_out.dims;

    boost::add_edge(vertex_in_id, vertex_out_id, edge_property, graph_);
    edge_param_map_[edge_param.id] = edge_param;
  }
}

template <class T>
void GraphEngine<T>::Instantiate() {
  node_map_.clear();
  edge_map_.clear();
  output_node_ptr_ = nullptr;
  // Instantiate the outputnode object;
  output_node_ptr_ = NodeFactory<T, OutputNodeSPtr<T>>::Instantiate(
      node_param_map_[output_node_id_]);
  node_map_[output_node_id_] = output_node_ptr_;
  // Instantiate node objects;
  for (auto node_pair : node_param_map_) {
    if (node_pair.first != output_node_id_) {
      NodeSPtr<T> node_ptr = NodeFactory<T, NodeSPtr<T>>::Instantiate(
          node_pair.second);
      node_map_[node_pair.first] = node_ptr;
    }
  }
  // Instantiate edge objects;
  for (auto edge_pair : edge_param_map_) {
    EdgeSPtr<T> edge_ptr = EdgeFactory<T>::Instantiate(edge_pair.second);
    edge_map_[edge_pair.first] = edge_ptr;
  }
}

// Instantiate class, otherwise compilation will fail
template class GraphEngine<float>;
template class GraphEngine<double>;
}  // intellgraph