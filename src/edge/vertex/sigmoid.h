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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_SIGMOID_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_SIGMOID_H_

#include "glog/logging.h"
#include "src/edge/vertex/op_vertex_impl.h"
#include "src/eigen.h"

namespace intellgraph {

class Sigmoid {
public:
  Sigmoid() = default;

  template <typename T> static void Activate(OpVertex<T> &vertex) {
    // Sigmoid activation function:
    // $\sigma(z)=1.0/(1.0+e^{-z})$
    MatrixX<T> *const activation = vertex.mutable_activation();
    activation->array() = 0.5 * (1.0 + Eigen::tanh(0.5 * activation->array()));
  }

  template <typename T> static void Derive(OpVertex<T> &vertex) {
    // Derivative equation:
    // $d\sigma/dz=\sigma(z)(1-\sigma(z))$
    MatrixX<T> *const activation = vertex.mutable_activation();
    activation->array() *= (1.0 - activation->array());
  }

protected:
  ~Sigmoid() = default;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_SIGMOID_H_
