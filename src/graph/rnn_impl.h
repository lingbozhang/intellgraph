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
#ifndef INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_
#define INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_

#include <map>
#include <set>

#include "src/edge.h"
#include "src/edge/dense_edge_impl.h"
#include "src/edge/input_vertex.h"
#include "src/edge/output_vertex.h"
#include "src/graph.h"
#include "src/proto/edge_parameter.pb.h"
#include "src/proto/vertex_parameter.pb.h"
#include "src/solver.h"
#include "src/visitor.h"
#include "src/visitor/backward_visitor.h"
#include "src/visitor/forward_visitor.h"
#include "src/visitor/init_vertex_visitor.h"
#include "src/visitor/resize_vertex_visitor.h"

namespace intellgraph {

template <typename T> class RnnImpl : public Graph {
public:
  RnnImpl(int sequence_length, std::unique_ptr<Visitor<T>> graph_init_visitor,
          std::unique_ptr<Solver<T>> solver,
          const typename Graph::AdjacencyList &adj_list, int input_vertex_id,
          int output_vertex_id, const std::map<int, int> &state_out_by_state_in,
          const std::set<VertexParameter> &vertex_params,
          const std::set<EdgeParameter> &edge_params);
  ~RnnImpl() = default;

private:
  int sequence_length_;
  std::unique_ptr<Visitor<T>> init_visitor_;
  std::unique_ptr<Solver<T>> solver_;

  // Graph topology
  const typename Graph::AdjacencyList adjacency_list_;
  std::vector<int> topological_order_;

  // Graph data
  InputVertex<T> *input_vertex_ = nullptr;
  OutputVertex<T> *output_vertex_ = nullptr;
  std::map<int, int> state_out_by_state_in_;
  std::map<int, std::unique_ptr<OpVertex<T>>> vertex_by_id_;
  std::map<int, std::unique_ptr<Edge<T>>> edge_by_id_;

  // Graph Visitors
  ResizeVertexVisitor<T> resize_vertex_visitor_;
  InitVertexVisitor<T> init_vtx_visitor_;
  BackwardVisitor<T> backward_visitor_;
  ForwardVisitor<T> forward_visitor_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_
