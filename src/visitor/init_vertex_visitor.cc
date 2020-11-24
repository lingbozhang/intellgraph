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
#include "src/visitor/init_vertex_visitor.h"

#include "glog/logging.h"
#include "src/edge/dense_edge_impl.h"
#include "src/eigen.h"

namespace intellgraph {

template <typename T> InitVertexVisitor<T>::InitVertexVisitor() = default;

template <typename T> InitVertexVisitor<T>::~InitVertexVisitor() = default;

template <typename T>
void InitVertexVisitor<T>::Visit(
    DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) {
  LOG(INFO) << "OpVertex " << edge.vertex_out()->id()
            << " is zero initialized.";

  OpVertex<T> *const vtx_out = edge.vertex_out();
  vtx_out->mutable_activation()->setZero();
  vtx_out->mutable_delta()->setZero();
}

// Explicit instantiation
template class InitVertexVisitor<float>;
template class InitVertexVisitor<double>;

} // namespace intellgraph

