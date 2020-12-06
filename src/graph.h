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
#ifndef INTELLGRAPH_SRC_GRAPH_GRAPH_H_
#define INTELLGRAPH_SRC_GRAPH_GRAPH_H_

#include <memory>

#include "boost/graph/topological_sort.hpp"
#include "glog/logging.h"
#include "src/boost.h"
#include "src/edge.h"
#include "src/eigen.h"
#include "src/proto/edge_parameter.pb.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/solver.h"
#include "src/visitor.h"

namespace intellgraph {

template <typename T> class Graph {
public:
  Graph(const ::google::protobuf::RepeatedPtrField<::intellgraph::EdgeParameter>
            &edge_params) {
    for (const auto &edge_param : edge_params) {
      EdgeProperty edge_property;
      edge_property.id = edge_param.id();
      VertexDescriptor v_in_id = edge_param.vertex_in_id();
      VertexDescriptor v_out_id = edge_param.vertex_out_id();
      if (!boost::add_edge(v_in_id, v_out_id, edge_property, adjacency_list_)
               .second) {
        LOG(ERROR) << "Add edge failed: edge " << edge_param.id()
                   << " has already been added into the adjacency list!";
      }
    }
    // Determines Forward orders by topological sorting
    topological_sort(adjacency_list_, std::back_inserter(topological_order_));
  }
  virtual ~Graph() = default;

  // Traverses the graph
  template <class Visitor>
  void Traverse(Visitor &visitor,
                const std::map<int, std::unique_ptr<Edge<T>>> &edge_by_id) {
    for (auto it = topological_order_.rbegin(); it != topological_order_.rend();
         ++it) {
      int vtx_id = *it;
      AdjacencyList::out_edge_iterator edge_it, edge_it_end;
      for (std::tie(edge_it, edge_it_end) = out_edges(vtx_id, adjacency_list_);
           edge_it != edge_it_end; ++edge_it) {
        int edge_id = adjacency_list_[*edge_it].id;
        edge_by_id.at(edge_id)->Accept(visitor);
      }
    }
  }

  // Traverses the graph reversely
  template <class Visitor>
  void RTraverse(Visitor &visitor,
                 const std::map<int, std::unique_ptr<Edge<T>>> &edge_by_id) {
    for (int vtx_id : topological_order_) {
      AdjacencyList::in_edge_iterator edge_it, edge_it_end;
      for (std::tie(edge_it, edge_it_end) = in_edges(vtx_id, adjacency_list_);
           edge_it != edge_it_end; ++edge_it) {
        int edge_id = adjacency_list_[*edge_it].id;
        edge_by_id.at(edge_id)->Accept(visitor);
      }
    }
  }

  virtual void Initialize(Visitor<T> &init_visitor) = 0;
  virtual void Train(const MatrixX<T> &feature,
                     const Eigen::Ref<const MatrixX<int>> &labels) = 0;
  virtual T CalculateLoss(const MatrixX<T> &test_feature,
                          const MatrixX<int> &test_labels) = 0;
  virtual void SetSolver(std::unique_ptr<Solver<T>> solver) = 0;

private:
  // Graph topology
  AdjacencyList adjacency_list_;
  std::vector<int> topological_order_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_GRAPH_H_

