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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_CROSS_ENTROPY_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_CROSS_ENTROPY_H_

#include <limits>

#include "glog/logging.h"
#include "src/edge/vertex/output_vertex_impl.h"
#include "src/edge/vertex/sigmoid.h"
#include "src/eigen.h"

namespace intellgraph {

class CrossEntropy {
public:
  CrossEntropy() = default;

  template <typename T>
  static void Activate(OutputVertexImpl<T, CrossEntropy> &vertex) {
    // Sigmoid activation function:
    // $\sigma(z)=1.0/(1.0+e^{-z})$
    Sigmoid::Activate(vertex);
  }

  template <typename T>
  static void Derive(OutputVertexImpl<T, CrossEntropy> &vertex) {
    // Derivative equation:
    // $d\sigma/dz=\sigma(z)(1-\sigma(z))$
    Sigmoid::Derive(vertex);
  }

  template <typename T>
  static T CalcLoss(OutputVertexImpl<T, CrossEntropy> &vertex,
                    const Eigen::Ref<const MatrixX<T>> &labels) {
    DCHECK_EQ(vertex.row(), labels.rows());
    DCHECK_EQ(vertex.col(), labels.cols());

    const Eigen::Map<const MatrixX<T>> &act = vertex.act();
    // Type epsilon is added inside the log function to avoid overflow
    T epsilon = std::numeric_limits<T>::epsilon();
    T loss = (labels.array() * (epsilon + act.array()).log() +
              (1.0 - labels.array()) * (1.0 - act.array() + epsilon).log())
                 .sum();
    int batch_size = vertex.col();
    return -loss / batch_size;
  }

  template <typename T>
  static void CalcDelta(OutputVertexImpl<T, CrossEntropy> &vertex,
                        const Eigen::Ref<const MatrixX<T>> &labels) {
    DCHECK_EQ(vertex.row(), labels.rows());
    DCHECK_EQ(vertex.col(), labels.cols());

    Eigen::Map<MatrixX<T>> delta = vertex.mutable_delta();
    const Eigen::Map<const MatrixX<T>> &act = vertex.act();

    delta = act - labels;
  }

protected:
  ~CrossEntropy() = default;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_CROSS_ENTROPY_H_
