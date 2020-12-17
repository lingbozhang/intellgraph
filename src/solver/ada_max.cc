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
#include "src/solver/ada_max.h"

#include <algorithm>
#include <cmath>

#include "glog/logging.h"

namespace intellgraph {

template <typename T>
AdaMax<T>::AdaMax(T eta, T lambda, T beta1, T beta2)
    : eta_(eta), lambda_(lambda), beta1_(beta1), beta2_(beta2) {
  DCHECK_GT(eta_, 0);
  DCHECK_GE(lambda_, 0);
  DCHECK(beta1_ > 0 && beta1_ < 1);
  DCHECK(beta2_ > 0 && beta2_ < 1);
}

template <typename T> AdaMax<T>::~AdaMax() = default;

template <typename T> void AdaMax<T>::Visit(Edge<T> &edge) {
  LOG(INFO) << "Edge " << edge.id() << " is updated with the Adam.";

  ++iteration_count_;

  Eigen::Map<MatrixX<T>> bias = edge.mutable_bias();
  Eigen::Map<MatrixX<T>> weight = edge.mutable_weight();

  MatrixX<T> nabla_weight;
  nabla_weight.noalias() = edge.CalcNablaWeight() + lambda_ * weight;
  const MatrixX<T> nabla_bias = edge.CalcNablaBias();

  Eigen::Map<MatrixX<T>> weight_first_moment = edge.mutable_weight_stores(0);
  Eigen::Map<MatrixX<T>> bias_first_moment = edge.mutable_bias_stores(0);
  Eigen::Map<MatrixX<T>> weight_ut = edge.mutable_weight_stores(1);
  Eigen::Map<MatrixX<T>> bias_ut = edge.mutable_bias_stores(1);

  // Updates first moments
  weight_first_moment.noalias() =
      beta1_ * weight_first_moment + (1.0 - beta1_) * nabla_weight;
  bias_first_moment.noalias() =
      beta1_ * bias_first_moment + (1.0 - beta1_) * nabla_bias;

  // Updates |weight_ut| and |bias_ut|
  DCHECK_EQ(weight_ut.cols(), bias_ut.rows());
  for (int col = 0; col < weight_ut.cols(); ++col) {
    for (int row = 0; row < weight_ut.rows(); ++row) {
      weight_ut(row, col) = std::max(beta2_ * weight_ut(row, col),
                                     std::abs(nabla_weight(row, col)));
    }
    bias_ut(col) = std::max(beta2_ * bias_ut(col), std::abs(nabla_bias(col)));
  }

  double first_moment_factor = 1.0 - std::pow(beta1_, iteration_count_);

  // Updates |weight| matrix
  weight.array() -= (eta_ / first_moment_factor) * weight_first_moment.array() /
                    weight_ut.array();

  // Updates |bias| vector
  bias.array() -= (eta_ / first_moment_factor) * bias_first_moment.array() /
                  bias_ut.array();
}

// Explicit instantiation
template class AdaMax<float>;
template class AdaMax<double>;

} // namespace intellgraph
