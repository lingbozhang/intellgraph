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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_INPUT_VERTEX_IMPL_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_INPUT_VERTEX_IMPL_H_

#include <memory>

#include "src/edge/input_vertex.h"
#include "src/eigen.h"
#include "src/proto/vertex_parameter.pb.h"

namespace intellgraph {

template <typename T, class Transformer>
class InputVertexImpl : public InputVertex<T> {
public:
  typedef T value_type;

  InputVertexImpl(int id, int row, int col);
  InputVertexImpl(const VertexParameter &vertex_param, int batch_size);
  ~InputVertexImpl() override;

  int id() const override;
  int row() const override;
  int col() const override;

  const Eigen::Block<const MatrixX<T>> activation() const override;
  void set_feature(const MatrixX<T> *feature) override;

private:
  int id_;
  int row_;
  int col_;

  const MatrixX<T> *feature_ = nullptr;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_INPUT_VERTEX_IMPL_H_
