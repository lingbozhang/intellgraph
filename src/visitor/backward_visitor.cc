/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
        Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#include "src/visitor/backward_visitor.h"

#include "glog/logging.h"
#include "src/edge/dense_edge_impl.h"
#include "src/eigen.h"

namespace intellgraph {

template <typename T> BackwardVisitor<T>::BackwardVisitor() = default;
template <typename T> BackwardVisitor<T>::~BackwardVisitor() = default;

template <typename T>
void BackwardVisitor<T>::Visit(
    DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) {
  LOG(INFO) << "DenseEdge " << edge.id() << " is backwarded.";

  OpVertex<T> *vtx_in = edge.vertex_in();
  OpVertex<T> *vtx_out = edge.vertex_out();

  const Eigen::Map<const MatrixX<T>> &act_in = vtx_in->act();
  const Eigen::Map<const MatrixX<T>> &weight = edge.weight();

  Eigen::Map<MatrixX<T>> delta_in = vtx_in->mutable_delta();
  Eigen::Map<MatrixX<T>> delta_out = vtx_out->mutable_delta();

  // Calculates |delta_in|:
  // $\delta^l= \mathcal{D}[f^\prime(z^l)]W^{l+1}\delta^{l+1}$
  // Delta matrix data are updated rather than overwritten
  if (delta_in.data()) {
    vtx_in->Derive();
    delta_in.array() += (weight * delta_out).array() * act_in.array();
  }
}

// Explicit instantiation
template class BackwardVisitor<float>;
template class BackwardVisitor<double>;

} // namespace intellgraph
