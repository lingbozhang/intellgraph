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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_OUTPUT_VERTEX_IMPL_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_OUTPUT_VERTEX_IMPL_H_

#include "glog/logging.h"
#include "src/edge/output_vertex.h"
#include "src/proto/vertex_parameter.pb.h"

namespace intellgraph {

template <typename T, class Algorithm>
class OutputVertexImpl : public Algorithm, public OutputVertex<T> {
public:
  typedef T value_type;

  OutputVertexImpl(int id, int row, int col);
  OutputVertexImpl(const VertexParameter &vtx_param, int batch_size);
  ~OutputVertexImpl() override;

  void Activate() override;
  void Derive() override;
  void ResizeVertex(int batch_size) override;

  int id() const override;
  int row() const override;
  int col() const override;

  const MatrixX<T> &activation() const override;
  MatrixX<T> *mutable_activation() override;
  MatrixX<T> *mutable_delta() override;
  VectorX<T> *mutable_bias() override;

  T CalcLoss(const Eigen::Ref<const MatrixX<T>> &labels) override;
  T CalcAccuracy(const Eigen::Ref<const MatrixX<T>> &labels) override;
  void CalcDelta(const Eigen::Ref<const MatrixX<T>> &labels) override;

private:
  int id_;
  int row_;
  int col_;

  std::unique_ptr<MatrixX<T>> activation_;
  std::unique_ptr<MatrixX<T>> delta_;
  std::unique_ptr<VectorX<T>> bias_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_OUTPUT_VERTEX_IMPL_H_
