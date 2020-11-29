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
#include "src/edge/vertex/output_vertex_impl.h"

#include "glog/logging.h"
#include "src/edge/vertex/cross_entropy.h"
#include "src/edge/vertex/sigmoid_l2.h"

namespace intellgraph {

template <typename T, class Algorithm>
OutputVertexImpl<T, Algorithm>::OutputVertexImpl(int id, int row, int col)
    : id_(id), row_(row), col_(col) {
  DCHECK_GE(id_, 0);
  DCHECK_GT(row_, 0);
  DCHECK_GT(col_, 0);

  activation_ = std::make_unique<MatrixX<T>>(row_, col_);
  delta_ = std::make_unique<MatrixX<T>>(row_, col_);
  bias_ = std::make_unique<VectorX<T>>(row_);

  // Zero initialization
  activation_->setZero();
  delta_->setZero();
  bias_->setZero();
}

template <typename T, class Algorithm>
OutputVertexImpl<T, Algorithm>::OutputVertexImpl(
    const VertexParameter &vtx_param, int batch_size)
    : OutputVertexImpl(vtx_param.id(), vtx_param.dims(), batch_size) {}

template <typename T, class Algorithm>
OutputVertexImpl<T, Algorithm>::~OutputVertexImpl() = default;

template <typename T, class Algorithm>
void OutputVertexImpl<T, Algorithm>::Activate() {
  LOG(INFO) << "OutputVertexImpl " << id_ << " is activated.";
  Algorithm::Activate(*this);
}

template <typename T, class Algorithm>
void OutputVertexImpl<T, Algorithm>::Derive() {
  LOG(INFO) << "OutputVertexImpl " << id_ << " is derived.";
  Algorithm::Derive(*this);
}

template <typename T, class Algorithm>
void OutputVertexImpl<T, Algorithm>::ResizeVertex(int batch_size) {
  DCHECK(batch_size != col_);

  LOG(INFO) << "OutputVertexImpl " << id_ << " batch size is resized from "
            << col_ << " to " << batch_size;
  col_ = batch_size;
  activation_ = std::make_unique<MatrixX<T>>(row_, batch_size);
  delta_ = std::make_unique<MatrixX<T>>(row_, batch_size);
  // Zero initialization
  activation_->setZero();
  delta_->setZero();
}

template <typename T, class Algorithm>
int OutputVertexImpl<T, Algorithm>::id() const {
  return id_;
}

template <typename T, class Algorithm>
int OutputVertexImpl<T, Algorithm>::row() const {
  return row_;
}

template <typename T, class Algorithm>
int OutputVertexImpl<T, Algorithm>::col() const {
  return col_;
}

template <typename T, class Algorithm>
const MatrixX<T> &OutputVertexImpl<T, Algorithm>::activation() const {
  return *activation_;
}

template <typename T, class Algorithm>
MatrixX<T> *OutputVertexImpl<T, Algorithm>::mutable_activation() {
  return activation_.get();
}

template <typename T, class Algorithm>
VectorX<T> *OutputVertexImpl<T, Algorithm>::mutable_bias() {
  return bias_.get();
}

template <typename T, class Algorithm>
MatrixX<T> *OutputVertexImpl<T, Algorithm>::mutable_delta() {
  return delta_.get();
}

template <typename T, class Algorithm>
T OutputVertexImpl<T, Algorithm>::CalcLoss(
    const Eigen::Ref<const MatrixX<T>> &labels) {
  return Algorithm::CalcLoss(*this, labels);
}

template <typename T, class Algorithm>
void OutputVertexImpl<T, Algorithm>::CalcDelta(
    const Eigen::Ref<const MatrixX<T>> &labels) {
  Algorithm::CalcDelta(*this, labels);
}

// Explicit instantiation
template class OutputVertexImpl<float, SigmoidL2>;
template class OutputVertexImpl<double, SigmoidL2>;
template class OutputVertexImpl<float, CrossEntropy>;
template class OutputVertexImpl<double, CrossEntropy>;

} // namespace intellgraph
