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
#include "src/visitor.h"

namespace intellgraph {

template <typename T, class VertexIn, class VertexOut = VertexIn>
class DenseEdgeImpl : public Edge<T> {
public:
  explicit DenseEdgeImpl(int id, VertexIn *vtx_in, VertexOut *vtx_out);
  ~DenseEdgeImpl();

  void Accept(Visitor<T> &visitor) override { visitor.Visit(*this); }

  int id() const override;
  const MatrixX<T> &weight() override;
  MatrixX<T> *mutable_weight() override;
  MatrixX<T> *mutable_nabla_weight() override;

  VertexIn *const vertex_in();
  VertexOut *const vertex_out();

  const MatrixX<T> CalcNablaWeight() override;

private:
  int id_;
  VertexIn *const vtx_in_;
  VertexOut *const vtx_out_;

  std::unique_ptr<MatrixX<T>> weight_;
  std::unique_ptr<MatrixX<T>> nabla_weight_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class DenseEdgeImpl<float, OpVertex<float>>;
extern template class DenseEdgeImpl<double, OpVertex<double>>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_DENSE_EDGE_IMPL_H_
