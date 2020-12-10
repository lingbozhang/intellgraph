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
  DCHECK_GT(eta_, 0.0);
  DCHECK_GE(lambda_, 0.0);
}

template <typename T>
SgdSolver<T>::SgdSolver(const SolverConfig &config)
    : eta_(config.eta()), lambda_(config.lambda()) {
  DCHECK_GT(eta_, 0.0);
  DCHECK_GE(lambda_, 0.0);
}

template <typename T> SgdSolver<T>::~SgdSolver() = default;

template <typename T> void SgdSolver<T>::Visit(Edge<T> &edge) {
  LOG(INFO) << "DenseEdge " << edge.id() << " is updated with the SGD solver.";

  VectorX<T> *const bias = edge.mutable_bias();
  MatrixX<T> *const weight = edge.mutable_weight();

  const MatrixX<T> nabla_weight = edge.CalcNablaWeight();
  const MatrixX<T> &delta = edge.delta();

  // Updates |weight| matrix
  weight->array() =
      (1.0 - eta_ * lambda_) * weight->array() - eta_ * nabla_weight.array();

  // Updates |bias| vector
  bias->array() -= (eta_ / delta.cols()) * delta.rowwise().sum().array();
}

// Explicitly instantiation
template class SgdSolver<float>;
template class SgdSolver<double>;

} // namespace intellgraph
