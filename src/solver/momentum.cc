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
Momentum<T>::Momentum(T eta, T gama, T lambda)
    : eta_(eta), gama_(gama), lambda_(lambda) {
  DCHECK_GT(eta_, 0.0);
  DCHECK_GE(gama_, 0);
  DCHECK_LT(gama_, 1);
  DCHECK_GE(lambda_, 0.0);
}

template <typename T> Momentum<T>::~Momentum() = default;

template <typename T> void Momentum<T>::Visit(Edge<T> &edge) {
  LOG(INFO) << "Edge " << edge.id() << " is updated with the Momentum.";

  Eigen::Map<MatrixX<T>> bias = edge.mutable_bias();
  Eigen::Map<MatrixX<T>> weight = edge.mutable_weight();

  const MatrixX<T> nabla_weight = edge.CalcNablaWeight();
  const MatrixX<T> nabla_bias = edge.CalcNablaBias();

  Eigen::Map<MatrixX<T>> weight_moment = edge.mutable_weight_store_1();
  Eigen::Map<MatrixX<T>> bias_moment = edge.mutable_bias_store_1();

  // Updates the Moment
  weight_moment.array() =
      gama_ * weight_moment.array() +
      eta_ * (nabla_weight.array() + lambda_ * weight.array());
  bias_moment.array() = gama_ * bias_moment.array() + eta_ * nabla_bias.array();

  // Updates |weight| matrix
  weight.noalias() -= weight_moment;

  // Updates |bias| vector
  bias.noalias() -= bias_moment;
}

// Explicitly instantiation
template class Momentum<float>;
template class Momentum<double>;

} /* intellgraph */ 
