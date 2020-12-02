/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
        Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#include "src/edge/vertex/seq_vertex_impl.h"

#include "src/edge/vertex/relu.h"
#include "src/edge/vertex/sigmoid.h"

namespace intellgraph {

template <typename T, class Algorithm>
SeqVertexImpl<T, Algorithm>::SeqVertexImpl(int id, int row, int col)
    : op_vertex_(id, row, col) {}

template <typename T, class Algorithm>
SeqVertexImpl<T, Algorithm>::SeqVertexImpl(const VertexParameter &vtx_param,
                                           int time_length)
    : op_vertex_(vtx_param, time_length) {}

template <typename T, class Algorithm>
void SeqVertexImpl<T, Algorithm>::ForwardTimeByOneStep() {
  ++timestamp_;
}

template <typename T, class Algorithm>
int SeqVertexImpl<T, Algorithm>::GetCurrentTimeStep() const {
  return timestamp_;
}

template class SeqVertexImpl<float, Relu>;
template class SeqVertexImpl<double, Relu>;
template class SeqVertexImpl<float, Sigmoid>;
template class SeqVertexImpl<double, Sigmoid>;

} // namespace intellgraph
