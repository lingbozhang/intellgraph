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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_OP_VERTEX_IMPL_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_OP_VERTEX_IMPL_H_

#include <memory>

#include "src/edge/op_vertex.h"
#include "src/eigen.h"
#include "src/proto/vertex_parameter.pb.h"
#include "src/tensor/dyn_matrix.h"

namespace intellgraph {

// The class accepts a class template |Algorithm| and delegate function
// implementations such as Activate and Derive to the Algorithm class.
template <typename T, class Algorithm>
class OpVertexImpl : public Algorithm, public OpVertex<T> {
public:
  typedef T value_type;

  explicit OpVertexImpl(int id, int row, int col);
  explicit OpVertexImpl(const VertexParameter &vtx_param, int batch_size);
  ~OpVertexImpl() override;

  void Activate() override;
  void Derive() override;
  void ResizeVertex(int length) override;

  int id() const override;
  int row() const override;
  int col() const override;

  const Eigen::Map<const MatrixX<T>> &act() const override;
  Eigen::Map<MatrixX<T>> mutable_act() override;
  Eigen::Map<MatrixX<T>> mutable_delta() override;
  Eigen::Map<MatrixX<T>> mutable_bias() override;
  const MatrixX<T> CalcNablaBias() override;

private:
  int id_;
  int row_;
  int col_;

  DynMatrix<T> act_;
  DynMatrix<T> delta_;
  DynMatrix<T> bias_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_OP_VERTEX_IMPL_H_
