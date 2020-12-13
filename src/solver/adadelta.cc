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
#include "src/solver/adadelta.h"

#include "glog/logging.h"

namespace intellgraph {

template <typename T>
Adadelta<T>::Adadelta(T gamma, T lambda, T epsilon)
    : gamma_(gamma), lambda_(lambda), epsilon_(epsilon) {
  DCHECK(gamma_ > 0 && gamma_ < 1);
  DCHECK_GE(lambda_, 0);
  DCHECK_GT(epsilon_, 0);
}

template <typename T> Adadelta<T>::~Adadelta() = default;

template <typename T> void Adadelta<T>::Visit(Edge<T> &edge) {
  LOG(INFO) << "Edge " << edge.id() << " is updated with the Adadetla.";

  Eigen::Map<MatrixX<T>> bias = edge.mutable_bias();
  Eigen::Map<MatrixX<T>> weight = edge.mutable_weight();

  MatrixX<T> nabla_weight;
  nabla_weight.noalias() = edge.CalcNablaWeight() + lambda_ * weight;
  const MatrixX<T> nabla_bias = edge.CalcNablaBias();

  Eigen::Map<MatrixX<T>> g_mean = edge.mutable_weight_stores(0);
  Eigen::Map<MatrixX<T>> g_bias_mean = edge.mutable_bias_stores(0);
  Eigen::Map<MatrixX<T>> weight_update_square_mean =
      edge.mutable_weight_stores(1);
  Eigen::Map<MatrixX<T>> bias_update_square_mean = edge.mutable_bias_stores(1);

  // Updates |g_mean| and |g_bias_mean|
  g_mean.array() =
      gamma_ * g_mean.array() + (1.0 - gamma_) * nabla_weight.array().square();
  g_bias_mean.array() = gamma_ * g_bias_mean.array() +
                        (1.0 - gamma_) * nabla_bias.array().square();

  // Calcuates weight and bias updates
  MatrixX<T> weight_update =
      (weight_update_square_mean.array() + epsilon_).sqrt() *
      nabla_weight.array() / (g_mean.array() + epsilon_).sqrt();
  MatrixX<T> bias_update = (bias_update_square_mean.array() + epsilon_).sqrt() *
                           nabla_bias.array() /
                           (g_bias_mean.array() + epsilon_).sqrt();

  // Updates |weight| and |bias|
  weight.noalias() -= weight_update;
  bias.noalias() -= bias_update;

  // Updates weight and bias updates
  weight_update_square_mean.array() =
      gamma_ * weight_update_square_mean.array() +
      (1.0 - gamma_) * weight_update.array().square();
  bias_update_square_mean.array() =
      gamma_ * bias_update_square_mean.array() +
      (1.0 - gamma_) * bias_update.array().square();
}

// Explicit instantiation
template class Adadelta<float>;
template class Adadelta<double>;

} // namespace intellgraph
