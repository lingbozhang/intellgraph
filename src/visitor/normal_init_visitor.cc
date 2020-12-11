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
#include "src/visitor/normal_init_visitor.h"

#include <functional>
#include <math.h>

#include "glog/logging.h"
#include "src/edge/dense_edge_impl.h"
#include "src/utility/random.h"

namespace intellgraph {

template <typename T> NormalInitVisitor<T>::NormalInitVisitor() = default;

template <typename T> NormalInitVisitor<T>::~NormalInitVisitor() = default;

template <typename T>
void NormalInitVisitor<T>::Visit(
    DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) {
  LOG(INFO) << "DenseEdge " << edge.id() << " and "
            << "OpVertex " << edge.vertex_out()->id()
            << " are initialized with the Normal Distribution function";

  Eigen::Map<MatrixX<T>> weight = edge.mutable_weight();

  weight.array() = weight.array().unaryExpr(std::function<T(T)>(
      NormalFunctor<T>(0.0, std::sqrt(2.0 / weight.cols()))));
}

// Explicit instantiation
template class NormalInitVisitor<float>;
template class NormalInitVisitor<double>;

} // namespace intellgraph
