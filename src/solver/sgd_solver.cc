/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 3.0 (the "License");
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
#include "src/solver/sgd_solver.h"

#include "glog/logging.h"
#include "src/edge/dense_edge_impl.h"
#include "src/eigen.h"

namespace intellgraph {

template <typename T>
SgdSolver<T>::SgdSolver(T eta, T lambda) : eta_(eta), lambda_(lambda) {
  DCHECK_GT(eta, 0.0);
  DCHECK_GE(lambda, 0.0);
}

template <typename T> SgdSolver<T>::~SgdSolver() = default;

template <typename T>
void SgdSolver<T>::Visit(DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) {
  LOG(INFO) << "DenseEdge " << edge.id() << " is updated with the SGD solver.";

  OpVertex<T> *const vtx_out = edge.vertex_out();

  VectorX<T> *const bias_out = vtx_out->mutable_bias();
  MatrixX<T> *const delta_out = vtx_out->mutable_delta();

  MatrixX<T> *const weight = edge.mutable_weight();
  const MatrixX<T> *const nabla_weight = edge.mutable_nabla_weight();

  // Updates |weight| matrix
  weight->array() =
      (1.0 - eta_ * lambda_) * weight->array() - eta_ * nabla_weight->array();

  // Updates |bias| vector
  bias_out->array() -=
      (eta_ / delta_out->cols()) * delta_out->rowwise().sum().array();
}

// Explicitly instantiation
template class SgdSolver<float>;
template class SgdSolver<double>;

} // namespace intellgraph
