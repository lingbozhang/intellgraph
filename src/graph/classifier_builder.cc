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
#include "src/graph/classifier_builder.h"

#include "glog/logging.h"
#include "src/graph/classifier_impl.h"
#include "src/visitor/normal_init_visitor.h"

namespace intellgraph {

template <typename T> ClassifierBuilder<T>::ClassifierBuilder() = default;
template <typename T> ClassifierBuilder<T>::~ClassifierBuilder() = default;

template <typename T>
ClassifierBuilder<T> &
ClassifierBuilder<T>::AddEdge(const std::string &edge_type,
                              const VertexParameter &vtx_param_in,
                              const VertexParameter &vtx_param_out) {
  DCHECK(!edge_type.empty());
  graph_builder_.add_edge(edge_type, vtx_param_in, vtx_param_out);
  return *this;
}

template <typename T>
ClassifierBuilder<T> &ClassifierBuilder<T>::SetInputVertexId(int id) {
  DCHECK_GE(id, 0);
  graph_builder_.set_input_vertex_id(id);
  return *this;
}

template <typename T>
ClassifierBuilder<T> &ClassifierBuilder<T>::SetOutputVertexId(int id) {
  DCHECK_GE(id, 0);
  graph_builder_.set_output_vertex_id(id);
  return *this;
}

template <typename T>
ClassifierBuilder<T> &
ClassifierBuilder<T>::SetInitVisitor(std::unique_ptr<Visitor<T>> init_visitor) {
  DCHECK(init_visitor);
  init_visitor_ = std::move(init_visitor);
  return *this;
}

template <typename T>
ClassifierBuilder<T> &
ClassifierBuilder<T>::SetSolver(std::unique_ptr<Solver<T>> solver) {
  DCHECK(solver);
  solver_ = std::move(solver);
  return *this;
}

template <typename T>
std::unique_ptr<ClassifierImpl<T>> ClassifierBuilder<T>::Build() {
  if (graph_builder_.input_vertex_id() == -1) {
    LOG(ERROR) << "Build graph failed: the input vertex ID must be set!";
    return nullptr;
  }
  if (graph_builder_.output_vertex_id() == -1) {
    LOG(ERROR) << "Build graph failed: the output vertex ID must be set!";
    return nullptr;
  }
  if (!solver_) {
    LOG(ERROR) << "Build graph: solver is not set!";
    return nullptr;
  }
  if (!init_visitor_) {
    LOG(WARNING) << "Build graph: graph initialization visitor is not set, "
                    "initialize the graph with normal distribution!";
    init_visitor_ = std::make_unique<NormalInitVisitor<T>>();
  }
  return std::make_unique<ClassifierImpl<T>>(
      /*batch_size=*/1, std::move(init_visitor_), std::move(solver_),
      graph_builder_.adjacency_list(), graph_builder_.input_vertex_id(),
      graph_builder_.output_vertex_id(), graph_builder_.vertex_params(),
      graph_builder_.edge_params());
}

// Explicit instantiation
template class ClassifierBuilder<float>;
template class ClassifierBuilder<double>;

} // namespace intellgraph
