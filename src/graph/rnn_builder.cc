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
#include "src/graph/rnn_builder.h"

#include "glog/logging.h"

namespace intellgraph {

template <typename T> RnnBuilder<T>::RnnBuilder() = default;
template <typename T> RnnBuilder<T>::~RnnBuilder() = default;

template <typename T>
RnnBuilder<T> &RnnBuilder<T>::AddEdge(const std::string &edge_type,
                                      const VertexParameter &vtx_param_in,
                                      const VertexParameter &vtx_param_out) {
  DCHECK(!edge_type.empty());
  graph_builder_.add_edge(edge_type, vtx_param_in, vtx_param_out);
  return *this;
}

template <typename T>
RnnBuilder<T> &RnnBuilder<T>::AddStateVertexPair(int state_in, int state_out) {
  DCHECK_GE(state_in, 0);
  DCHECK_GE(state_out, 0);
  DCHECK_NE(state_in, state_out);

  if (state_out_by_state_in_.find(state_in) != state_out_by_state_in_.end()) {
    LOG(ERROR) << "Failed to add state vertex pair (" << state_in << ", "
               << state_out << "), state vertices have already been added.";
    return;
  }
  state_out_by_state_in_.insert(state_in, state_out);
}

template <typename T> RnnBuilder<T> &RnnBuilder<T>::SetInputVertexId(int id) {
  DCHECK_GE(id, 0);
  graph_builder_.set_input_vertex_id(id);
  return *this;
}

template <typename T> RnnBuilder<T> &RnnBuilder<T>::SetOutputVertexId(int id) {
  DCHECK_GE(id, 0);
  graph_builder_.set_output_vertex_id(id);
  return *this;
}

template <typename T>
RnnBuilder<T> &
RnnBuilder<T>::SetInitVisitor(std::unique_ptr<Visitor<T>> init_visitor) {
  DCHECK(init_visitor);
  init_visitor_ = std::move(init_visitor);
  return *this;
}

template <typename T>
RnnBuilder<T> &RnnBuilder<T>::SetSolver(std::unique_ptr<Solver<T>> solver) {
  DCHECK(solver);
  solver_ = std::move(solver);
  return *this;
}

} // namespace intellgraph
