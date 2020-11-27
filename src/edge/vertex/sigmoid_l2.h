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
#include "src/eigen.h"

namespace intellgraph {

class SigmoidL2 {
public:
  SigmoidL2() = default;

  template <typename T>
  static void Activate(OutputVertexImpl<T, SigmoidL2> &vertex) {
    // Sigmoid activation function:
    // $\sigma(z)=1.0/(1.0+e^{-z})$
    LOG(INFO) << "OutputVertex " << vertex.id()
              << " is activated with the Sigmoid function.";
    MatrixX<T> *activation = vertex.mutable_activation();
    for (size_t i = 0; i < activation->rows(); ++i) {
      for (size_t j = 0; j < activation->cols(); ++j) {
        T element_value = activation->array()(i, j);
        if (element_value >= 0.0) {
          activation->array()(i, j) = 1.0 / (1.0 + std::exp(-element_value));
        } else {
          activation->array()(i, j) =
              std::exp(element_value) / (1.0 + std::exp(element_value));
        }
      }
    }
  }

  template <typename T>
  static void Derive(OutputVertexImpl<T, SigmoidL2> &vertex) {
    // Derivative equation:
    // $d\sigma/dz=\sigma(z)(1-\sigma(z))$
    LOG(INFO) << "OutputVertex " << vertex.id()
              << " is derived with the Sigmoid function.";
    MatrixX<T> *activation = vertex.mutable_activation();
    activation->array() *= (1.0 - activation->array());
  }

  template <typename T>
  static T CalcLoss(OutputVertexImpl<T, SigmoidL2> &vertex,
                    const Eigen::Ref<const MatrixX<T>> &labels) {
    DCHECK_EQ(vertex.row(), labels.rows());
    DCHECK_EQ(vertex.col(), labels.cols());

    LOG(INFO) << "OutputVertex " << vertex.id() << " calculates L2-NORM loss.";
    const MatrixX<T> &activation = vertex.activation();
    T loss = (activation - labels.matrix()).squaredNorm();
    int batch_size = vertex.col();
    return loss / 2.0 / batch_size;
  }

  template <typename T>
  static void CalcDelta(OutputVertexImpl<T, SigmoidL2> &vertex,
                        const Eigen::Ref<const MatrixX<T>> &labels) {
    DCHECK_EQ(vertex.row(), labels.rows());
    DCHECK_EQ(vertex.col(), labels.cols());

    LOG(INFO) << "OutputVertex " << vertex.id()
              << " calculates delta based on the Sigmoid function";
    MatrixX<T> *delta = vertex.mutable_delta();
    const MatrixX<T> &activation = vertex.activation();

    delta->matrix() = activation - labels.matrix();
    vertex.Derive();
    delta->array() *= activation.array();
  }

protected:
  ~SigmoidL2() = default;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_SIGMOID_L2_H_
