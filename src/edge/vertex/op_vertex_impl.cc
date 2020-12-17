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
#include "src/edge/vertex/op_vertex_impl.h"

#include "glog/logging.h"
#include "src/edge/vertex/relu.h"
#include "src/edge/vertex/sigmoid.h"

namespace intellgraph {

template <typename T, class Algorithm>
OpVertexImpl<T, Algorithm>::OpVertexImpl(int id, int row, int col)
    : id_(id), row_(row), col_(col) {
  DCHECK_GE(id_, 0);
  DCHECK_GT(row_, 0);
  DCHECK_GT(col_, 0);

  act_ = DynMatrix<T>(row_, col_);
  delta_ = DynMatrix<T>(row_, col_);
  bias_ = DynMatrix<T>(row_, 1);
}

template <typename T, class Algorithm>
OpVertexImpl<T, Algorithm>::OpVertexImpl(const VertexParameter &vtx_param,
                                         int batch_size)
    : OpVertexImpl(vtx_param.id(), vtx_param.dims(), batch_size) {}

template <typename T, class Algorithm>
OpVertexImpl<T, Algorithm>::~OpVertexImpl() = default;

template <typename T, class Algorithm>
void OpVertexImpl<T, Algorithm>::Activate() {
  LOG(INFO) << "OpVertexImpl " << id_ << " is activated.";
  Algorithm::Activate(*this);
}

template <typename T, class Algorithm>
void OpVertexImpl<T, Algorithm>::Derive() {
  LOG(INFO) << "OpVertexImpl " << id_ << " is derived.";
  Algorithm::Derive(*this);
}

template <typename T, class Algorithm>
void OpVertexImpl<T, Algorithm>::ResizeVertex(int length) {
  DCHECK(length != col_);

  col_ = length;
  act_.Resize(row_, col_);
  delta_.Resize(row_, col_);
}

template <typename T, class Algorithm>
int OpVertexImpl<T, Algorithm>::id() const {
  return id_;
}

template <typename T, class Algorithm>
int OpVertexImpl<T, Algorithm>::row() const {
  return row_;
}

template <typename T, class Algorithm>
int OpVertexImpl<T, Algorithm>::col() const {
  return col_;
}

template <typename T, class Algorithm>
const Eigen::Map<const MatrixX<T>> &OpVertexImpl<T, Algorithm>::act() const {
  return act_.map();
}

template <typename T, class Algorithm>
Eigen::Map<MatrixX<T>> OpVertexImpl<T, Algorithm>::mutable_act() {
  return act_.mutable_map();
}

template <typename T, class Algorithm>
Eigen::Map<MatrixX<T>> OpVertexImpl<T, Algorithm>::mutable_delta() {
  return delta_.mutable_map();
}

template <typename T, class Algorithm>
Eigen::Map<MatrixX<T>> OpVertexImpl<T, Algorithm>::mutable_bias() {
  return bias_.mutable_map();
}

template <typename T, class Algorithm>
const MatrixX<T> OpVertexImpl<T, Algorithm>::CalcNablaBias() {
  return delta_.mutable_map().rowwise().sum() / col_;
}

// Explicit instantiation
template class OpVertexImpl<float, Relu>;
template class OpVertexImpl<double, Relu>;
template class OpVertexImpl<float, Sigmoid>;
template class OpVertexImpl<double, Sigmoid>;

} // namespace intellgraph
