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
#include "src/solver/adam.h"

#include <cmath>

#include "glog/logging.h"

namespace intellgraph {

template <typename T>
Adam<T>::Adam(T eta, T lambda, T beta1, T beta2, T epsilon)
    : eta_(eta), lambda_(lambda), beta1_(beta1), beta2_(beta2),
      epsilon_(epsilon) {
  DCHECK_GT(eta_, 0);
  DCHECK_GE(lambda_, 0);
  DCHECK(beta1_ > 0 && beta1_ < 1);
  DCHECK(beta2_ > 0 && beta2_ < 1);
  DCHECK_GT(epsilon_, 0);
}

template <typename T> Adam<T>::~Adam() = default;

template <typename T> void Adam<T>::Visit(Edge<T> &edge) {
  LOG(INFO) << "Edge " << edge.id() << " is updated with the Adam.";

  ++iteration_count_;

  Eigen::Map<MatrixX<T>> bias = edge.mutable_bias();
  Eigen::Map<MatrixX<T>> weight = edge.mutable_weight();

  MatrixX<T> nabla_weight;
  nabla_weight.noalias() = edge.CalcNablaWeight() + lambda_ * weight;
  const MatrixX<T> nabla_bias = edge.CalcNablaBias();

  Eigen::Map<MatrixX<T>> weight_first_moment = edge.mutable_weight_stores(0);
  Eigen::Map<MatrixX<T>> bias_first_moment = edge.mutable_bias_stores(0);
  Eigen::Map<MatrixX<T>> weight_second_moment = edge.mutable_weight_stores(1);
  Eigen::Map<MatrixX<T>> bias_second_moment = edge.mutable_bias_stores(1);

  // Updates first and second moments
  weight_first_moment.noalias() =
      beta1_ * weight_first_moment + (1.0 - beta1_) * nabla_weight;
  weight_second_moment.array() = beta2_ * weight_second_moment.array() +
                                 (1.0 - beta2_) * nabla_weight.array().square();
  bias_first_moment.noalias() =
      beta1_ * bias_first_moment + (1.0 - beta1_) * nabla_bias;
  bias_second_moment.array() = beta2_ * bias_second_moment.array() +
                               (1.0 - beta2_) * nabla_bias.array().square();

  double first_moment_factor = 1.0 - std::pow(beta1_, iteration_count_);
  double second_moment_factor = 1.0 - std::pow(beta2_, iteration_count_);
  // Updates |weight| matrix
  weight.array() -=
      (eta_ / first_moment_factor) * weight_first_moment.array() /
      ((weight_second_moment.array() / second_moment_factor).sqrt() + epsilon_);

  // Updates |bias| vector
  bias.array() -=
      (eta_ / first_moment_factor) * bias_first_moment.array() /
      ((bias_second_moment.array() / second_moment_factor).sqrt() + epsilon_);
}

// Explicit instantiation
template class Adam<float>;
template class Adam<double>;

} // namespace intellgraph

