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
#include "src/graph/classifier_impl.h"

#include "glog/logging.h"
#include "src/edge/factory.h"

namespace intellgraph {

template <typename T>
ClassifierImpl<T>::ClassifierImpl(
    int batch_size, std::unique_ptr<Visitor<T>> init_visitor,
    std::unique_ptr<Solver<T>> solver,
    const typename Graph::AdjacencyList &adjacency_list, int input_vertex_id,
    int output_vertex_id, const std::set<VertexParameter> &vertex_params,
    const std::set<EdgeParameter> &edge_params)
    : batch_size_(batch_size), init_visitor_(std::move(init_visitor)),
      solver_(std::move(solver)), adjacency_list_(adjacency_list),
      resize_vertex_visitor_(batch_size_) {
  DCHECK_GT(batch_size_, 0);
  DCHECK(init_visitor_);
  DCHECK(solver_);

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
      // Builds the default |threshold_|
      int dims = vertex_param.dims();
      threshold_ = MatrixX<T>(dims, 1);
      threshold_.setConstant(0.5);
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
  this->Traverse(*init_visitor_);
}

template <typename T> ClassifierImpl<T>::~ClassifierImpl() = default;

template <typename T>
void ClassifierImpl<T>::Train(const MatrixX<T> &feature,
                              const Eigen::Ref<const MatrixX<int>> &labels) {
  DCHECK_GT(labels.cols(), 0);

  this->Forward(feature);
  this->Backward(labels);
  this->Traverse(*solver_);
}

template <typename T>
T ClassifierImpl<T>::CalculateLoss(const MatrixX<T> &test_feature,
                                   const MatrixX<int> &test_labels) {
  this->Forward(test_feature);
  return output_vertex_->CalcLoss(test_labels.cast<T>());
}

template <typename T>
const MatrixX<T>
ClassifierImpl<T>::GetProbabilityDist(const MatrixX<T> &feature) {
  this->Forward(feature);
  return output_vertex_->activation().leftCols(batch_size_);
}

template <typename T>
void ClassifierImpl<T>::SetSolver(std::unique_ptr<Solver<T>> solver) {
  DCHECK(solver);
  solver_ = std::move(solver);
}

template <typename T>
void ClassifierImpl<T>::SetThreshold(const MatrixX<T> &threshold) {
  DCHECK_EQ(threshold.rows(), threshold_.rows());
  threshold_ = threshold;
}

template <typename T>
const MatrixX<T> ClassifierImpl<T>::CalcConfusionMatrix(
    const MatrixX<T> &test_feature,
    const Eigen::Ref<const MatrixX<int>> &test_labels) {
  DCHECK_EQ(test_feature.cols(), test_labels.cols());
  DCHECK_EQ(output_vertex_->row(), test_labels.rows());

  this->Forward(test_feature);
  const MatrixX<T> &activation = output_vertex_->activation();
  int class_num = activation.rows() == 1 ? 2 : activation.rows();
  int batch_size = output_vertex_->col();
  MatrixX<T> confusion_matrix = MatrixX<T>::Zero(class_num, class_num);

  if (activation.rows() == 1) {
    // Binary classification
    int true_positive = 0;
    int false_negative = 0;
    int false_positive = 0;
    int true_negative = 0;

    MatrixX<int> predication = MatrixX<int>::Zero(1, batch_size);
    predication = (activation.leftCols(batch_size).array() >
                   threshold_.array().replicate(1, batch_size))
                      .template cast<int>();
    int correct_predication =
        (predication.leftCols(batch_size).array() == test_labels.array())
            .count();
    true_positive = (predication.array() * test_labels.array()).count();
    true_negative = correct_predication - true_positive;
    int positive = test_labels.count();
    int negative = batch_size - positive;
    false_negative = positive - true_positive;
    false_positive = negative - true_negative;

    confusion_matrix << true_negative, false_negative, false_positive,
        true_positive;
  } else {
    // Multi-class classification
    MatrixX<T> weighted_probability =
        activation.leftCols(batch_size).array() *
        threshold_.array().replicate(activation.rows(), batch_size);
    for (int i = 0; i < batch_size; ++i) {
      int predicted_class, actual_class;
      weighted_probability.col(i).maxCoeff(&predicted_class);
      test_labels.col(i).maxCoeff(&actual_class);
      confusion_matrix(predicted_class, actual_class)++;
    }
  }
  return confusion_matrix;
}

template <typename T>
OpVertex<T> *ClassifierImpl<T>::mutable_vertex(int vertex_id) {
  if (vertex_by_id_.find(vertex_id) != vertex_by_id_.end()) {
    return vertex_by_id_.at(vertex_id).get();
  }
  LOG(WARNING) << "Vertex " << vertex_id << " does not exist in the graph.";
  return nullptr;
}

template <typename T> Edge<T> *ClassifierImpl<T>::mutable_edge(int edge_id) {
  if (edge_by_id_.find(edge_id) != edge_by_id_.end()) {
    return edge_by_id_.at(edge_id).get();
  }
  LOG(WARNING) << "Edge " << edge_id << " does not exist in the graph.";
  return nullptr;
}

template <typename T> void ClassifierImpl<T>::ZeroInitializeVertex() {
  this->Traverse(init_vtx_visitor_);
}

template <typename T>
void ClassifierImpl<T>::Forward(const MatrixX<T> &feature) {
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
void ClassifierImpl<T>::Backward(const Eigen::Ref<const MatrixX<int>> &labels) {
  output_vertex_->CalcDelta(labels.cast<T>());
  this->ReverseTraverse(backward_visitor_);
}

template <typename T>
template <class Visitor>
void ClassifierImpl<T>::Traverse(Visitor &visitor) {
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
void ClassifierImpl<T>::ReverseTraverse(Visitor &visitor) {
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
template class ClassifierImpl<float>;
template class ClassifierImpl<double>;

} // namespace intellgraph
