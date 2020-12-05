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
#ifndef INTELLGRAPH_SRC_GRAPH_RNN_BUILDER_H_
#define INTELLGRAPH_SRC_GRAPH_RNN_BUILDER_H_

#include <map>
#include <string>

#include "src/graph/graph_builder.h"

namespace intellgraph {

template <typename T> class RnnBuilder {
public:
  RnnBuilder();
  ~RnnBuilder();

  RnnBuilder &AddEdge(const std::string &edge_type,
                      const VertexParameter &vtx_param_in,
                      const VertexParameter &vtx_param_out);
  RnnBuilder &AddStateVertexPair(int state_in, int state_out);

  RnnBuilder &SetInputVertexId(int id);
  RnnBuilder &SetOutputVertexId(int id);
  RnnBuilder &SetInitVisitor(std::unique_ptr<Visitor<T>> init_visitor);
  RnnBuilder &SetSolver(std::unique_ptr<Solver<T>> solver);

private:
  GraphBuilder<T> graph_builder_;
  std::map<int, int> state_out_by_state_in_;
  std::unique_ptr<Visitor<T>> init_visitor_;
  std::unique_ptr<Solver<T>> solver_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_RNN_BUILDER_H_
