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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_INPUT_VERTEX_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_INPUT_VERTEX_H_

#include "src/edge/op_vertex.h"
#include "src/eigen.h"
#include "src/logging.h"

namespace intellgraph {

struct DummyTransformer {};

template <typename T>
class InputVertex : public OpVertex<T> {
public:
  typedef T value_type;

  InputVertex() = default;
  ~InputVertex() override = default;

  // Dummy implementations:
  void Activate() override {}
  void Derive() override {}
  void ResizeVertex(int length) override {}
  Eigen::Map<MatrixX<T>> mutable_act() override {
    NOTREACHED();
    return Eigen::Map<MatrixX<T>>(nullptr, -1, -1);
  }
  Eigen::Map<MatrixX<T>> mutable_delta() override {
    return Eigen::Map<MatrixX<T>>(nullptr, -1, -1);
  }
  Eigen::Map<MatrixX<T>> mutable_bias() override {
    NOTREACHED();
    return Eigen::Map<MatrixX<T>>(nullptr, -1, -1);
  }

  virtual void set_feature(const MatrixX<T> *feature) = 0;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_INPUT_VERTEX_H_
