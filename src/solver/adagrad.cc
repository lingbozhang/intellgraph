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
#include "src/solver/adagrad.h"

#include "glog/logging.h"

namespace intellgraph {

template <typename T>
Adagrad<T>::Adagrad(T eta, T lambda, T epsilon)
    : eta_(eta), lambda_(lambda), epsilon_(epsilon) {
  DCHECK_GT(eta_, 0);
  DCHECK_GE(lambda_, 0);
  DCHECK_GT(epsilon, 0);
}

template <typename T> Adagrad<T>::~Adagrad() = default;

template <typename T> void Adagrad<T>::Visit(Edge<T> &edge) {
  LOG(INFO) << "Edge " << edge.id() << " is updated with the Adagrad.";

  Eigen::Map<MatrixX<T>> bias = edge.mutable_bias();
  Eigen::Map<MatrixX<T>> weight = edge.mutable_weight();

  const MatrixX<T> nabla_weight = edge.CalcNablaWeight();
  const MatrixX<T> nabla_bias = edge.CalcNablaBias();

  Eigen::Map<MatrixX<T>> g = edge.mutable_weight_stores(0);
  Eigen::Map<MatrixX<T>> g_bias = edge.mutable_bias_stores(0);

  g.array() += (nabla_weight.array() + lambda_ * weight.array()).square();
  g_bias.array() += nabla_bias.array().square();

  // Updates |weight| matrix
  weight.array() -= eta_ * nabla_weight.array() / (g.array() + epsilon_).sqrt();

  // Updates |bias| vector
  bias.array() -=
      eta_ * nabla_bias.array() / (g_bias.array() + epsilon_).sqrt();
}

// Explicit instantiation
template class Adagrad<float>;
template class Adagrad<double>;

} // namespace intellgraph
