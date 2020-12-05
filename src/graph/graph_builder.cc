/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-1.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
        Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#include "src/graph/graph_builder.h"

#include "glog/logging.h"

namespace intellgraph {

static bool operator<(const VertexParameter &left,
                      const VertexParameter &right) {
  return left.id() < right.id();
}

static bool operator<(const EdgeParameter &left, const EdgeParameter &right) {
  return left.id() < right.id();
}

template <typename T> GraphBuilder<T>::GraphBuilder() = default;
template <typename T> GraphBuilder<T>::~GraphBuilder() = default;

template <typename T>
void GraphBuilder<T>::add_edge(const std::string &edge_type,
                               const VertexParameter &vtx_param_in,
                               const VertexParameter &vtx_param_out) {
  DCHECK(!edge_type.empty());

  vertex_params_.insert(vtx_param_in);
  vertex_params_.insert(vtx_param_out);

  // Constructs the edge parameter
  EdgeParameter edge_param;
  int edge_id = edge_params_.size();
  edge_param.set_id(edge_id);
  edge_param.set_type(edge_type);
  int vtx_in_id = vtx_param_in.id();
  int vtx_out_id = vtx_param_out.id();
  edge_param.set_vertex_in_id(vtx_in_id);
  edge_param.set_vertex_out_id(vtx_out_id);
  edge_params_.insert(edge_param);

  // Stores the edge information into the adjacency list
  Graph::EdgeProperty edge_property{edge_id};
  Graph::VertexDescriptor v_in_id = vtx_in_id;
  Graph::VertexDescriptor v_out_id = vtx_out_id;
  if (!boost::add_edge(v_in_id, v_out_id, edge_property, adjacency_list_)
           .second) {
    LOG(ERROR) << "Add edge failed: edge " << edge_id
               << " has already been added into the adjacency list!";
  }
}

template <typename T>
const std::set<VertexParameter> &GraphBuilder<T>::vertex_params() {
  DCHECK_GT(vertex_params_.size(), 0);
  return vertex_params_;
}

template <typename T>
const std::set<EdgeParameter> &GraphBuilder<T>::edge_params() {
  DCHECK_GT(edge_params_.size(), 0);
  return edge_params_;
}

template <typename T>
const Graph::AdjacencyList &GraphBuilder<T>::adjacency_list() {
  return adjacency_list_;
}

template <typename T> void GraphBuilder<T>::set_input_vertex_id(int id) {
  DCHECK_GE(id, 0);
  input_vertex_id_ = id;
}

template <typename T> int GraphBuilder<T>::input_vertex_id() {
  return input_vertex_id_;
}

template <typename T> void GraphBuilder<T>::set_output_vertex_id(int id) {
  DCHECK_GE(id, 0);
  output_vertex_id_ = id;
}

template <typename T> int GraphBuilder<T>::output_vertex_id() {
  return output_vertex_id_;
}

// Explicit instantiation
template class GraphBuilder<float>;
template class GraphBuilder<double>;

} // namespace intellgraph
