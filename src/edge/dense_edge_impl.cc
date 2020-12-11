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
#include "src/tensor/dyn_matrix.h"
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

  weight_ = DynMatrix<T>(row_, col_);
  // Initialization
  weight_.mutable_map().array() = weight_.mutable_map().array().unaryExpr(
      std::function<T(T)>(NormalFunctor<T>(0.0, std::sqrt(2.0 / col_))));
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
const Eigen::Map<const MatrixX<T>> &
DenseEdgeImpl<T, VertexIn, VertexOut>::weight() {
  return weight_.map();
};

template <typename T, class VertexIn, class VertexOut>
Eigen::Map<MatrixX<T>> DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_weight() {
  return weight_.mutable_map();
};

template <typename T, class VertexIn, class VertexOut>
Eigen::Map<MatrixX<T>> DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_bias() {
  return vtx_out_->mutable_bias();
};

template <typename T, class VertexIn, class VertexOut>
Eigen::Map<MatrixX<T>>
DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_weight_store_1() {
  // Lazy initialization
  if (!weight_store_1_.data()) {
    weight_store_1_ = DynMatrix<T>(row_, col_);
  }
  return weight_store_1_.mutable_map();
}

template <typename T, class VertexIn, class VertexOut>
Eigen::Map<MatrixX<T>>
DenseEdgeImpl<T, VertexIn, VertexOut>::mutable_bias_store_1() {
  // Lazy initialization
  if (!bias_store_1_.data()) {
    bias_store_1_ = DynMatrix<T>(vtx_out_->mutable_bias().rows(),
                                 vtx_out_->mutable_bias().cols());
  }
  // Resizes the |moment_bias_| if dimensions mismatch with the corresponding
  // delta matrix
  if (bias_store_1_.row() != vtx_out_->mutable_bias().rows() ||
      bias_store_1_.col() != vtx_out_->mutable_bias().cols()) {
    bias_store_1_.Resize(vtx_out_->mutable_bias().rows(),
                         vtx_out_->mutable_bias().cols());
  }
  return bias_store_1_.mutable_map();
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
  return (vtx_in_->act() * vtx_out_->mutable_delta().transpose()) / batch_size;
}

template <typename T, class VertexIn, class VertexOut>
const MatrixX<T> DenseEdgeImpl<T, VertexIn, VertexOut>::CalcNablaBias() {
  return vtx_out_->mutable_delta().rowwise().sum() /
         vtx_out_->mutable_delta().cols();
}

// Explicitly instantiation
template class DenseEdgeImpl<float, OpVertex<float>>;
template class DenseEdgeImpl<double, OpVertex<double>>;

} // namespace intellgraph
