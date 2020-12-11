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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_SIGMOID_L2_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_SIGMOID_L2_H_

#include "glog/logging.h"
#include "src/edge/vertex/output_vertex_impl.h"
#include "src/edge/vertex/sigmoid.h"
#include "src/eigen.h"

namespace intellgraph {

class SigmoidL2 {
public:
  SigmoidL2() = default;

  template <typename T>
  static void Activate(OutputVertexImpl<T, SigmoidL2> &vertex) {
    // Sigmoid activation function:
    // $\sigma(z)=1.0/(1.0+e^{-z})$
    Sigmoid::Activate(vertex);
  }

  template <typename T>
  static void Derive(OutputVertexImpl<T, SigmoidL2> &vertex) {
    // Derivative equation:
    // $d\sigma/dz=\sigma(z)(1-\sigma(z))$
    Sigmoid::Derive(vertex);
  }

  template <typename T>
  static T CalcLoss(OutputVertexImpl<T, SigmoidL2> &vertex,
                    const Eigen::Ref<const MatrixX<T>> &labels) {
    DCHECK_EQ(vertex.row(), labels.rows());
    DCHECK_EQ(vertex.col(), labels.cols());

    const Eigen::Map<const MatrixX<T>> &act = vertex.act();
    T loss = (act - labels).squaredNorm();
    return loss / 2.0 / act.cols();
  }

  template <typename T>
  static void CalcDelta(OutputVertexImpl<T, SigmoidL2> &vertex,
                        const Eigen::Ref<const MatrixX<T>> &labels) {
    DCHECK_EQ(vertex.row(), labels.rows());
    DCHECK_EQ(vertex.col(), labels.cols());

    Eigen::Map<MatrixX<T>> delta = vertex.mutable_delta();
    const Eigen::Map<const MatrixX<T>> &act = vertex.act();

    delta.noalias() = act - labels;
    Sigmoid::Derive(vertex);
    delta.array() *= act.array();
  }

protected:
  ~SigmoidL2() = default;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_SIGMOID_L2_H_
