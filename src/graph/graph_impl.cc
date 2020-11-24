/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 3.0 (the "License");
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
#include "src/graph/graph_impl.h"

#include "glog/logging.h"
#include "src/edge/factory.h"

namespace intellgraph {

template <typename T>
GraphImpl<T>::GraphImpl(int batch_size,
                        std::unique_ptr<Visitor<T>> graph_init_visitor,
                        const typename Graph::AdjacencyList &adjacency_list,
                        int input_vertex_id, int output_vertex_id,
                        const std::set<VertexParameter> &vertex_params,
                        const std::set<EdgeParameter> &edge_params)
    : batch_size_(batch_size),
      graph_init_visitor_(std::move(graph_init_visitor)),
      adjacency_list_(adjacency_list), resize_vertex_visitor_(batch_size_) {
  DCHECK_GT(batch_size_, 0);
  DCHECK(graph_init_visitor_);

  // Instantiates vertices
  for (const auto &vertex_param : vertex_params) {
    int vtx_id = vertex_param.id();
    if (vtx_id == input_vertex_id) {
      // Instantiates the input vertex
      std::unique_ptr<InputVertex<T>> input_vertex =
          Factory::InstantiateVertex<InputVertex<T>>(vertex_param, batch_size_);
      input_vertex_ = input_vertex.get();
      vertex_by_id_.try_emplace(vtx_id, std::move(input_vertex));
    } else if (vtx_id == output_vertex_id) {
      // Instantiates intermediate vertices
      std::unique_ptr<OutputVertex<T>> output_vertex =
          Factory::InstantiateVertex<OutputVertex<T>>(vertex_param,
                                                      batch_size_);
      output_vertex_ = output_vertex.get();
      vertex_by_id_.try_emplace(vtx_id, std::move(output_vertex));
    } else {
      // Instantiates output vertices
      std::unique_ptr<OpVertex<T>> vertex =
          Factory::InstantiateVertex<OpVertex<T>>(vertex_param, batch_size_);
      vertex_by_id_.try_emplace(vtx_id, std::move(vertex));
    }
  }

  // Instantiates edges
  for (const auto &edge_param : edge_params) {
    int edge_id = edge_param.id();
    const std::string &edge_type = edge_param.type();
    int vtx_in_id = edge_param.vertex_in_id();
    int vtx_out_id = edge_param.vertex_out_id();

    edge_by_id_.try_emplace(
        edge_id, Factory::InstantiateEdge<Edge<T>, OpVertex<T>, OpVertex<T>>(
                     edge_type, edge_id, vertex_by_id_.at(vtx_in_id).get(),
                     vertex_by_id_.at(vtx_out_id).get()));
  }

  // Determines Forward orders by topological sorting
  topological_sort(adjacency_list_, std::back_inserter(topological_order_));
  // Initializes the graph
  this->Traverse(*graph_init_visitor_);
}

template <typename T> GraphImpl<T>::~GraphImpl() = default;

template <typename T>
template <class Visitor>
void GraphImpl<T>::Train(Visitor &solver, const MatrixX<T> &feature,
                         const MatrixX<T> &labels) {
  DCHECK_GT(labels.cols(), 0);

  this->Forward(feature);
  this->Backward(labels);
  this->Traverse(solver);
}
template <typename T>
const MatrixX<T> &GraphImpl<T>::Infer(const MatrixX<T> &feature) {
  this->Forward(feature);
  return *output_vertex_->mutable_activation();
}

template <typename T>
T GraphImpl<T>::CalculateLoss(const MatrixX<T> &test_feature,
                              const MatrixX<T> &test_labels) {
  this->Forward(test_feature);
  return output_vertex_->CalcLoss(test_labels);
}

template <typename T>
T GraphImpl<T>::CalculateAccuracy(const MatrixX<T> &test_feature,
                                  const MatrixX<T> &test_labels) {
  this->Forward(test_feature);
  return output_vertex_->CalcAccuracy(test_labels);
}

template <typename T> OpVertex<T> *GraphImpl<T>::mutable_vertex(int vertex_id) {
  if (vertex_by_id_.find(vertex_id) != vertex_by_id_.end()) {
    return vertex_by_id_.at(vertex_id).get();
  }
  LOG(WARNING) << "Vertex " << vertex_id << " does not exist in the graph.";
  return nullptr;
}

template <typename T> Edge<T> *GraphImpl<T>::mutable_edge(int edge_id) {
  if (edge_by_id_.find(edge_id) != edge_by_id_.end()) {
    return edge_by_id_.at(edge_id).get();
  }
  LOG(WARNING) << "Edge " << edge_id << " does not exist in the graph.";
  return nullptr;
}

template <typename T> void GraphImpl<T>::ZeroInitializeVertex() {
  this->Traverse(init_vtx_visitor_);
}

template <typename T> void GraphImpl<T>::Forward(const MatrixX<T> &feature) {
  if (batch_size_ != feature.cols()) {
    batch_size_ = feature.cols();
    resize_vertex_visitor_.set_batch_size(batch_size_);
    this->Traverse(resize_vertex_visitor_);
  }
  input_vertex_->set_feature(&feature);
  this->ZeroInitializeVertex();
  this->Traverse(forward_visitor_);
  output_vertex_->Activate();
}

template <typename T>
void GraphImpl<T>::Backward(const Eigen::Ref<const MatrixX<T>> &labels) {
  output_vertex_->CalcDelta(labels);
  this->ReverseTraverse(backward_visitor_);
}

template <typename T>
template <class Visitor>
void GraphImpl<T>::Traverse(Visitor &visitor) {
  // Trasverses the graph
  for (auto it = topological_order_.rbegin(); it != topological_order_.rend();
       ++it) {
    int vtx_id = *it;
    Graph::AdjacencyList::out_edge_iterator edge_it, edge_it_end;
    for (std::tie(edge_it, edge_it_end) = out_edges(vtx_id, adjacency_list_);
         edge_it != edge_it_end; ++edge_it) {
      int edge_id = adjacency_list_[*edge_it].id;
      edge_by_id_.at(edge_id)->Accept(visitor);
    }
  }
}

template <typename T>
template <class Visitor>
void GraphImpl<T>::ReverseTraverse(Visitor &visitor) {
  // Trasverses the graph reversely
  for (int vtx_id : topological_order_) {
    Graph::AdjacencyList::in_edge_iterator edge_it, edge_it_end;
    for (std::tie(edge_it, edge_it_end) = in_edges(vtx_id, adjacency_list_);
         edge_it != edge_it_end; ++edge_it) {
      int edge_id = adjacency_list_[*edge_it].id;
      edge_by_id_.at(edge_id)->Accept(visitor);
    }
  }
}

// Explicit instantiation
template class GraphImpl<float>;
template void GraphImpl<float>::Train<SgdSolver<float>>(SgdSolver<float> &,
                                                        const MatrixX<float> &,
                                                        const MatrixX<float> &);
template class GraphImpl<double>;
template void GraphImpl<double>::Train<SgdSolver<double>>(
    SgdSolver<double> &, const MatrixX<double> &, const MatrixX<double> &);
} // namespace intellgraph
