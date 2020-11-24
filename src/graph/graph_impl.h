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
#ifndef INTELLGRAPH_SRC_GRAPH_GRAPH_IMPL_H_
#define INTELLGRAPH_SRC_GRAPH_GRAPH_IMPL_H_

#include <memory>
#include <set>

#include "boost/graph/topological_sort.hpp"
#include "glog/logging.h"
#include "src/edge/dense_edge_impl.h"
#include "src/edge/input_vertex.h"
#include "src/edge/op_vertex.h"
#include "src/edge/output_vertex.h"
#include "src/eigen.h"
#include "src/graph.h"
#include "src/proto/edge_parameter.pb.h"
#include "src/proto/vertex_parameter.pb.h"
#include "src/solver/sgd_solver.h"
#include "src/visitor.h"
#include "src/visitor/backward_visitor.h"
#include "src/visitor/forward_visitor.h"
#include "src/visitor/init_vertex_visitor.h"
#include "src/visitor/resize_vertex_visitor.h"

namespace intellgraph {

template <typename T> class GraphImpl : public Graph {
public:
  GraphImpl(int batch_size, std::unique_ptr<Visitor<T>> graph_init_visitor,
            const typename Graph::AdjacencyList &adj_list, int input_vertex_id,
            int output_vertex_id,
            const std::set<VertexParameter> &vertex_params,
            const std::set<EdgeParameter> &edge_params);
  ~GraphImpl() override;

  template <class Visitor>
  void Train(Visitor &solver, const MatrixX<T> &feature,
             const MatrixX<T> &labels);
  const MatrixX<T> &Infer(const MatrixX<T> &feature);
  T CalculateLoss(const MatrixX<T> &test_feature,
                  const MatrixX<T> &test_labels);
  T CalculateAccuracy(const MatrixX<T> &test_feature,
                      const MatrixX<T> &test_labels);
  OpVertex<T> *mutable_vertex(int vertex_id);
  Edge<T> *mutable_edge(int edge_id);

private:
  void ZeroInitializeVertex();
  void Forward(const MatrixX<T> &feature);
  void Backward(const Eigen::Ref<const MatrixX<T>> &labels);

  template <class Visitor> void Traverse(Visitor &visitor);
  template <class Visitor> void ReverseTraverse(Visitor &visitor);

  int batch_size_ = 0;
  std::unique_ptr<Visitor<T>> graph_init_visitor_;
  // Graph topology
  const typename Graph::AdjacencyList adjacency_list_;
  std::vector<int> topological_order_;

  // Graph data
  InputVertex<T> *input_vertex_ = nullptr;
  OutputVertex<T> *output_vertex_ = nullptr;
  std::map<int, std::unique_ptr<OpVertex<T>>> vertex_by_id_;
  std::map<int, std::unique_ptr<Edge<T>>> edge_by_id_;

  // Graph Visitors
  ResizeVertexVisitor<T> resize_vertex_visitor_;
  InitVertexVisitor<T> init_vtx_visitor_;
  BackwardVisitor<T> backward_visitor_;
  ForwardVisitor<T> forward_visitor_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class GraphImpl<float>;
extern template void GraphImpl<float>::Train<SgdSolver<float>>(
    SgdSolver<float> &, const MatrixX<float> &, const MatrixX<float> &);
extern template class GraphImpl<double>;
extern template void GraphImpl<double>::Train<SgdSolver<double>>(
    SgdSolver<double> &, const MatrixX<double> &, const MatrixX<double> &);

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_GRAPH_IMPL_H_
