/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-1.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
        Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#ifndef INTELLGRAPH_SRC_EDGE_DENSE_EDGE_IMPL_H_
#define INTELLGRAPH_SRC_EDGE_DENSE_EDGE_IMPL_H_

#include <memory>

#include "src/edge.h"
#include "src/edge/op_vertex.h"
#include "src/eigen.h"
#include "src/solver.h"
#include "src/tensor/dyn_matrix.h"
#include "src/visitor.h"

namespace intellgraph {

template <typename T, class VertexIn, class VertexOut = VertexIn>
class DenseEdgeImpl : public Edge<T> {
public:
  explicit DenseEdgeImpl(int id, VertexIn *vtx_in, VertexOut *vtx_out);
  ~DenseEdgeImpl();

  void Accept(Visitor<T> &visitor) override { visitor.Visit(*this); }
  void Accept(Solver<T> &solver) override { solver.Visit(*this); }

  int id() const override;
  int row() const override;
  int col() const override;

  const Eigen::Map<const MatrixX<T>> &weight() override;
  Eigen::Map<MatrixX<T>> mutable_weight() override;
  Eigen::Map<MatrixX<T>> mutable_bias() override;
  Eigen::Map<MatrixX<T>> mutable_weight_store_1() override;
  Eigen::Map<MatrixX<T>> mutable_bias_store_1() override;

  const MatrixX<T> CalcNablaWeight() override;
  const MatrixX<T> CalcNablaBias() override;

  VertexIn *const vertex_in();
  VertexOut *const vertex_out();

private:
  int id_ = -1;
  int row_ = 0;
  int col_ = 0;

  VertexIn *const vtx_in_;
  VertexOut *const vtx_out_;

  DynMatrix<T> weight_;
  DynMatrix<T> weight_store_1_;
  DynMatrix<T> bias_store_1_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class DenseEdgeImpl<float, OpVertex<float>>;
extern template class DenseEdgeImpl<double, OpVertex<double>>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_DENSE_EDGE_IMPL_H_
