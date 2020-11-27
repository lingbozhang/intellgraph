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
#include "src/visitor/normal_init_visitor.h"

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
GraphBuilder<T> &
GraphBuilder<T>::AddEdge(const std::string &edge_type,
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
    return *this;
  }
  return *this;
}

template <typename T>
GraphBuilder<T> &GraphBuilder<T>::SetInputVertexId(int id) {
  DCHECK_GE(id, 0);

  input_vertex_id_ = id;
  return *this;
}

template <typename T>
GraphBuilder<T> &GraphBuilder<T>::SetOutputVertexId(int id) {
  DCHECK_GE(id, 0);

  output_vertex_id_ = id;
  return *this;
}

template <typename T>
GraphBuilder<T> &GraphBuilder<T>::SetBatchSize(int batch_size) {
  batch_size_ = batch_size;
  return *this;
}

template <typename T>
GraphBuilder<T> &GraphBuilder<T>::SetGraphInitVisitor(
    std::unique_ptr<Visitor<T>> graph_init_visitor) {
  DCHECK(graph_init_visitor);

  graph_init_visitor_ = std::move(graph_init_visitor);
  return *this;
}

template <typename T>
std::unique_ptr<ClassifierImpl<T>> GraphBuilder<T>::Build() {
  if (input_vertex_id_ == -1) {
    LOG(ERROR) << "Build graph failed: the input vertex ID must be set!";
    return nullptr;
  }
  if (output_vertex_id_ == -1) {
    LOG(ERROR) << "Build graph failed: the output vertex ID must be set!";
    return nullptr;
  }
  if (batch_size_ == 0) {
    LOG(ERROR) << "Build graph failed: the batch size must be set!";
    return nullptr;
  }
  if (!graph_init_visitor_) {
    LOG(WARNING) << "Build graph: graph initialization visitor is not set, "
                    "initialize the graph with normal distribution";
    graph_init_visitor_ = std::make_unique<NormalInitVisitor<T>>();
  }
  return std::make_unique<ClassifierImpl<T>>(
      batch_size_, std::move(graph_init_visitor_), adjacency_list_,
      input_vertex_id_, output_vertex_id_, vertex_params_, edge_params_);
}

// Explicit instantiation
template class GraphBuilder<float>;
template class GraphBuilder<double>;

} // namespace intellgraph
