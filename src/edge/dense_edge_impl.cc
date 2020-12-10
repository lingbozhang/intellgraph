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

#include <functional>
#include <math.h>

#include "glog/logging.h"
#include "src/utility/random.h"

namespace intellgraph {

template <typename T, class VertexIn, class VertexOut>
DenseEdgeImpl<T, VertexIn, VertexOut>::DenseEdgeImpl(int id, VertexIn *vtx_in,
                                                     VertexOut *vtx_out)
    : id_(id), row_(vtx_in->row()), col_(vtx_out->row()), vtx_in_(vtx_in),
      vtx_out_(vtx_out) {
  DCHECK_GE(id_, 0);
  DCHECK_GT(row_, 0);
  DCHECK_GT(col_, 0);
  DCHECK(vtx_in_);
  DCHECK(vtx_out_);
  DCHECK_EQ(vtx_in_->col(), vtx_out_->col());

  weight_ = std::make_unique<MatrixX<T>>(row_, col_);

  // Initialization
  weight_->array() = weight_->array().unaryExpr(std::function<T(T)>(
      NormalFunctor<T>(0.0, std::sqrt(2.0 / weight_->cols()))));
}

template <typename T, class VertexIn, class VertexOut>
DenseEdgeImpl<T, VertexIn, VertexOut>::~DenseEdgeImpl() = default;

template <typename T, class VertexIn, class VertexOut>
int DenseEdgeImpl<T, VertexIn, VertexOut>::id() const {
  return id_;
}

template <typename T, class VertexIn, class VertexOut>
int DenseEdgeImpl<T, VertexIn, VertexOut>::row() const {
  return row_;
}

template <typename T, class VertexIn, class VertexOut>
int DenseEdgeImpl<T, VertexIn, VertexOut>::col() const {
  return col_;
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
VectorX<T> *DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_bias() {
  return vtx_out_->mutable_bias();
};

template <typename T, class VertexIn, class VertexOut>
Eigen::Block<MatrixX<T>> DenseEdgeImpl<T, VertexIn, VertexOut>::delta() {
  return vtx_out_->mutable_delta()->block(0, 0, vtx_out_->row(),
                                          vtx_out_->col());
}

template <typename T, class VertexIn, class VertexOut>
MatrixX<T> *DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_moment() {
  // Lazy initialization
  if (!moment_) {
    moment_ = std::make_unique<MatrixX<T>>(row_, col_);
    moment_->setConstant(0);
  }
  return moment_.get();
}

template <typename T, class VertexIn, class VertexOut>
VectorX<T> *DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_moment_delta() {
  // Lazy initialization
  if (!moment_delta_) {
    moment_delta_ = std::make_unique<VectorX<T>>(col_);
    moment_delta_->setConstant(0);
  }
  return moment_delta_.get();
}

template <typename T, class VertexIn, class VertexOut>
VertexIn *const DenseEdgeImpl<T, VertexIn, VertexOut>::vertex_in() {
  return vtx_in_;
}

template <typename T, class VertexIn, class VertexOut>
VertexOut *const DenseEdgeImpl<T, VertexIn, VertexOut>::vertex_out() {
  return vtx_out_;
}

template <typename T, class VertexIn, class VertexOut>
const MatrixX<T> DenseEdgeImpl<T, VertexIn, VertexOut>::CalcNablaWeight() {
  // Calculates |nabla_weight|:
  // $\frac{\partial loss}{\partial W^l}=a^{l-1}(\delta^{l})^T$
  int batch_size = vtx_in_->col();
  return (vtx_in_->activation().leftCols(batch_size) *
          vtx_out_->mutable_delta()->leftCols(batch_size).transpose()) /
         batch_size;
}

// Explicitly instantiation
template class DenseEdgeImpl<float, OpVertex<float>>;
template class DenseEdgeImpl<double, OpVertex<double>>;

} // namespace intellgraph
