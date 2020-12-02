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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_SEQ_OUTPUT_IMPL_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_SEQ_OUTPUT_IMPL_H_

#include "glog/logging.h"
#include "src/edge/seq_output.h"
#include "src/edge/vertex/output_vertex_impl.h"
#include "src/proto/vertex_parameter.pb.h"

namespace intellgraph {

template <typename T, class Algorithm>
class SeqOutputImpl : public Algorithm, public SeqOutput<T> {
public:
  typedef T value_type;

  SeqOutputImpl(int id, int row, int col);
  SeqOutputImpl(const VertexParameter &vtx_param, int batch_size);
  ~SeqOutputImpl() override;

  void Activate() override { output_vertex_.Activate(); }
  void Derive() override { output_vertex_.Derive(); }
  void ResizeVertex(int length) override {
    output_vertex_.ResizeVertex(length);
  }

  int id() const override { return output_vertex_.id(); }
  int row() const override { return output_vertex_.row(); }
  int col() const override { return output_vertex_.col(); }

  const MatrixX<T> &activation() const override {
    return output_vertex_.activation();
  }
  MatrixX<T> *mutable_activation() override {
    return output_vertex_.mutable_activation();
  }
  MatrixX<T> *mutable_delta() override {
    return output_vertex_.mutable_delta();
  }
  VectorX<T> *mutable_bias() override { return output_vertex_.mutable_bias(); }

  T CalcLoss(const Eigen::Ref<const MatrixX<T>> &labels) override {
    return output_vertex_.CalcLoss(labels);
  }
  void CalcDelta(const Eigen::Ref<const MatrixX<T>> &labels) override {
    output_vertex_.CalcDelta(labels);
  }

  void ForwardTimeByOneStep() override;
  int GetCurrentTimeStep() const override;

private:
  OutputVertexImpl<T, Algorithm> output_vertex_;
  int timestamp_ = 0;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_SEQ_OUTPUT_IMPL_H_
