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
#include "src/edge/vertex/seq_output_impl.h"

#include "src/edge/vertex/cross_entropy.h"
#include "src/edge/vertex/sigmoid_l2.h"

namespace intellgraph {

template <typename T, class Algorithm>
SeqOutputImpl<T, Algorithm>::SeqOutputImpl(int id, int row, int col)
    : output_vertex_(id, row, col) {}

template <typename T, class Algorithm>
SeqOutputImpl<T, Algorithm>::SeqOutputImpl(const VertexParameter &vtx_param,
                                           int length)
    : output_vertex_(vtx_param, length) {}

template <typename T, class Algorithm>
void SeqOutputImpl<T, Algorithm>::Activate() {
  output_vertex_.Activate();
}

template <typename T, class Algorithm>
void SeqOutputImpl<T, Algorithm>::Derive() {
  output_vertex_.Derive();
}

template <typename T, class Algorithm>
void SeqOutputImpl<T, Algorithm>::ResizeVertex(int length) {
  output_vertex_.ResizeVertex(length);
}

template <typename T, class Algorithm>
int SeqOutputImpl<T, Algorithm>::id() const {
  return output_vertex_.id();
}

template <typename T, class Algorithm>
int SeqOutputImpl<T, Algorithm>::row() const {
  return output_vertex_.row();
}

template <typename T, class Algorithm>
int SeqOutputImpl<T, Algorithm>::col() const {
  return output_vertex_.col();
}

template <typename T, class Algorithm>
const MatrixX<T> &SeqOutputImpl<T, Algorithm>::activation() const {
  return output_vertex_.activation();
}

template <typename T, class Algorithm>
MatrixX<T> *SeqOutputImpl<T, Algorithm>::mutable_activation() {
  return output_vertex_.mutable_activation();
}

template <typename T, class Algorithm>
MatrixX<T> *SeqOutputImpl<T, Algorithm>::mutable_delta() {
  return output_vertex_.mutable_delta();
}

template <typename T, class Algorithm>
VectorX<T> *SeqOutputImpl<T, Algorithm>::mutable_bias() {
  return output_vertex_.mutable_bias();
}

template <typename T, class Algorithm>
T SeqOutputImpl<T, Algorithm>::CalcLoss(
    const Eigen::Ref<const MatrixX<T>> &labels) {
  return output_vertex_.CalcLoss(labels);
}

template <typename T, class Algorithm>
void SeqOutputImpl<T, Algorithm>::CalcDelta(
    const Eigen::Ref<const MatrixX<T>> &labels) {
  output_vertex_.CalcDelta(labels);
}

template <typename T, class Algorithm>
void SeqOutputImpl<T, Algorithm>::ForwardTimeByOneStep() {
  ++timestamp_;
}

template <typename T, class Algorithm>
int SeqOutputImpl<T, Algorithm>::GetCurrentTimeStep() const {
  return timestamp_;
}

// Explicit instantiation
template class SeqOutputImpl<float, SigmoidL2>;
template class SeqOutputImpl<double, SigmoidL2>;
template class SeqOutputImpl<float, CrossEntropy>;
template class SeqOutputImpl<double, CrossEntropy>;

} // namespace intellgraph
