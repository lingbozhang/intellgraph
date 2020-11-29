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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_RELU_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_RELU_H_

#include "glog/logging.h"
#include "src/edge/vertex/op_vertex_impl.h"
#include "src/eigen.h"

namespace intellgraph {

class Relu {
public:
  Relu() = default;

  template <typename T> static void Activate(OpVertexImpl<T, Relu> &vertex) {
    MatrixX<T> *const activation = vertex.mutable_activation();
    for (size_t i = 0; i < activation->rows(); ++i) {
      for (size_t j = 0; j < activation->cols(); ++j) {
        if (activation->matrix()(i, j) < 0) {
          activation->matrix()(i, j) = 0;
        }
      }
    }
  }

  template <typename T> static void Derive(OpVertexImpl<T, Relu> &vertex) {
    MatrixX<T> *const activation = vertex.mutable_activation();
    for (size_t i = 0; i < activation->rows(); ++i) {
      for (size_t j = 0; j < activation->cols(); ++j) {
        DCHECK_GE(activation->matrix()(i, j), 0);
        if (activation->matrix()(i, j) > 0) {
          activation->matrix()(i, j) = 1;
        }
      }
    }
  }

protected:
  ~Relu() = default;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_RELU_H_
