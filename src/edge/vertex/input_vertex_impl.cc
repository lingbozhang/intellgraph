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
#include "src/edge/vertex/input_vertex_impl.h"

#include "src/logging.h"

namespace intellgraph {

template <typename T, class Transformer>
InputVertexImpl<T, Transformer>::InputVertexImpl(int id, int row, int col)
    : id_(id), row_(row), col_(col) {
  DCHECK_GE(id_, 0);
  DCHECK_GT(row, 0);
  DCHECK_GT(col, 0);
}

template <typename T, class Transformer>
InputVertexImpl<T, Transformer>::InputVertexImpl(
    const VertexParameter &vtx_param, int batch_size)
    : InputVertexImpl(vtx_param.id(), vtx_param.dims(), batch_size) {}

template <typename T, class Transformer>
InputVertexImpl<T, Transformer>::~InputVertexImpl() = default;

template <typename T, class Transformer>
int InputVertexImpl<T, Transformer>::id() const {
  return id_;
}

template <typename T, class Transformer>
int InputVertexImpl<T, Transformer>::row() const {
  return row_;
}

template <typename T, class Transformer>
int InputVertexImpl<T, Transformer>::col() const {
  return col_;
}

template <typename T, class Transformer>
const Eigen::Map<const MatrixX<T>> &
InputVertexImpl<T, Transformer>::act() const {
  DCHECK(feature_);
  return feature_map_;
}

template <typename T, class Transformer>
void InputVertexImpl<T, Transformer>::set_feature(const MatrixX<T> *feature) {
  DCHECK(feature);
  DCHECK_EQ(row_, feature->rows());

  LOG(INFO) << "InputVertexImpl feeds a feature value.";
  if (col_ != feature->cols()) {
    col_ = feature->cols();
  }
  feature_ = feature;
  new (&feature_map_)
      Eigen::Map<const MatrixX<T>>(feature_->data(), row_, col_);
}

// Explicitly instantiation
template class InputVertexImpl<float, DummyTransformer>;
template class InputVertexImpl<double, DummyTransformer>;

} // namespace intellgraph
