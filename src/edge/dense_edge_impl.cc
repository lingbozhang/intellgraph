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
#include "src/edge/dense_edge_impl.h"

#include "glog/logging.h"

namespace intellgraph {

template <typename T, class VertexIn, class VertexOut>
DenseEdgeImpl<T, VertexIn, VertexOut>::DenseEdgeImpl(int id, VertexIn *vtx_in,
                                                     VertexOut *vtx_out)
    : id_(id), vtx_in_(vtx_in), vtx_out_(vtx_out) {
  DCHECK_GE(id_, 0);
  DCHECK(vtx_in_);
  DCHECK(vtx_out_);
  DCHECK_EQ(vtx_in_->col(), vtx_out_->col());

  int row = vtx_in_->row();
  int col = vtx_out_->row();
  DCHECK_GT(row, 0);
  DCHECK_GT(col, 0);

  weight_ = std::make_unique<MatrixX<T>>(row, col);
  nabla_weight_ = std::make_unique<MatrixX<T>>(row, col);

  // Initialization
  weight_->setIdentity();
  nabla_weight_->setZero();
}

template <typename T, class VertexIn, class VertexOut>
DenseEdgeImpl<T, VertexIn, VertexOut>::~DenseEdgeImpl() = default;

template <typename T, class VertexIn, class VertexOut>
int DenseEdgeImpl<T, VertexIn, VertexOut>::id() const {
  return id_;
}

template <typename T, class VertexIn, class VertexOut>
const MatrixX<T> &DenseEdgeImpl<T, VertexIn, VertexOut>::weight() {
  return *weight_;
};

template <typename T, class VertexIn, class VertexOut>
MatrixX<T> *DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_weight() {
  return weight_.get();
};

template <typename T, class VertexIn, class VertexOut>
MatrixX<T> *DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_nabla_weight() {
  return nabla_weight_.get();
}

template <typename T, class VertexIn, class VertexOut>
VertexIn *const DenseEdgeImpl<T, VertexIn, VertexOut>::vertex_in() {
  return vtx_in_;
}

template <typename T, class VertexIn, class VertexOut>
VertexOut *const DenseEdgeImpl<T, VertexIn, VertexOut>::vertex_out() {
  return vtx_out_;
}

// Explicitly instantiation
template class DenseEdgeImpl<float, OpVertex<float>>;
template class DenseEdgeImpl<double, OpVertex<double>>;

} // namespace intellgraph
