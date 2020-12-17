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
#ifndef INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_
#define INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_

#include <map>
#include <set>

#include "src/edge.h"
#include "src/edge/op_vertex.h"
#include "src/edge/output_vertex.h"
#include "src/edge/vertex/input_vertex.h"
#include "src/eigen.h"
#include "src/graph.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/solver.h"
#include "src/visitor.h"

namespace intellgraph {

template <typename T> class RnnImpl : public Graph<T> {
public:
  RnnImpl(const GraphParameter &graph_parameter);
  ~RnnImpl() override = default;

private:
  int sequence_length_;
  std::unique_ptr<Solver<T>> solver_;
  MatrixX<T> threshold_;
  InputVertex<T> *input_vertex_ = nullptr;
  OutputVertex<T> *output_vertex_ = nullptr;
  std::map<int, std::unique_ptr<OpVertex<T>>> vertex_by_id_;
  std::map<int, std::unique_ptr<Edge<T>>> edge_by_id_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_
