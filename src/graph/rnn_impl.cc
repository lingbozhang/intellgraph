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
#include "src/graph/rnn_impl.h"

#include "glog/logging.h"

namespace intellgraph {

template <typename T>
RnnImpl<T>::RnnImpl(int sequence_length,
                    std::unique_ptr<Visitor<T>> init_visitor,
                    std::unique_ptr<Solver<T>> solver,
                    const typename Graph::AdjacencyList &adj_list,
                    int input_vertex_id, int output_vertex_id,
                    const std::map<int, int> &state_out_by_state_in,
                    const std::set<VertexParameter> &vertex_params,
                    const std::set<EdgeParameter> &edge_params)
    : sequence_length_(sequence_length), init_visitor_(std::move(init_visitor)),
      solver_(std::move(solver)), adjacency_list_(adj_list),
      state_out_by_state_in_(state_out_by_state_in) {
  DCHECK_GT(sequence_length_, 0);
  DCHECK(init_visitor_);
  DCHECK(solver_);
}

} // namespace intellgraph
