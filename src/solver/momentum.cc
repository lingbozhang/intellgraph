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
#include "src/solver/momentum.h"

#include "glog/logging.h"
#include "src/edge/op_vertex.h"

namespace intellgraph{

template <typename T>
Momentum<T>::Momentum(T eta, T gamma, T lambda)
    : eta_(eta), gamma_(gamma), lambda_(lambda) {
  DCHECK_GT(eta_, 0.0);
  DCHECK(gamma_ > 0 && gamma_ < 1);
  DCHECK_GE(lambda_, 0.0);
}

template <typename T> Momentum<T>::~Momentum() = default;

template <typename T> void Momentum<T>::Visit(Edge<T> &edge) {
  LOG(INFO) << "Edge " << edge.id() << " is updated with the Momentum.";

  Eigen::Map<MatrixX<T>> bias = edge.mutable_bias();
  Eigen::Map<MatrixX<T>> weight = edge.mutable_weight();

  MatrixX<T> nabla_weight;
  nabla_weight.noalias() = edge.CalcNablaWeight() + lambda_ * weight;
  const MatrixX<T> nabla_bias = edge.CalcNablaBias();

  Eigen::Map<MatrixX<T>> weight_update = edge.mutable_weight_stores(0);
  Eigen::Map<MatrixX<T>> bias_update = edge.mutable_bias_stores(0);

  // Updates the Moment
  weight_update.noalias() = gamma_ * weight_update + eta_ * nabla_weight;
  bias_update.noalias() = gamma_ * bias_update + eta_ * nabla_bias;

  // Updates |weight| matrix
  weight.noalias() -= weight_update;

  // Updates |bias| vector
  bias.noalias() -= bias_update;
}

// Explicitly instantiation
template class Momentum<float>;
template class Momentum<double>;

} /* intellgraph */ 
