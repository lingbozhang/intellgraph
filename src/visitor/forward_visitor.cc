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
#include "src/visitor/forward_visitor.h"

#include "glog/logging.h"
#include "src/edge/dense_edge_impl.h"
#include "src/eigen.h"

namespace intellgraph {

template <typename T> ForwardVisitor<T>::ForwardVisitor() = default;
template <typename T> ForwardVisitor<T>::~ForwardVisitor() = default;

template <typename T>
void ForwardVisitor<T>::Visit(
    DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) {
  LOG(INFO) << "DenseEdge " << edge.id() << " is forwarded.";

  OpVertex<T> *vtx_in = edge.vertex_in();
  OpVertex<T> *vtx_out = edge.vertex_out();

  const MatrixX<T> &activation_in = vtx_in->activation();
  MatrixX<T> *const activation_out = vtx_out->mutable_activation();
  const VectorX<T> *const bias_out = vtx_out->mutable_bias();

  const MatrixX<T> *const weight = edge.mutable_weight();

  // Activation matrix data of the outbound vertex is updated rather than
  // overwritten
  vtx_in->Activate();
  activation_out->noalias() +=
      (weight->transpose() * activation_in.matrix()).colwise() + *bias_out;
}

// Explicit instantiation
template class ForwardVisitor<float>;
template class ForwardVisitor<double>;

} // namespace intellgraph