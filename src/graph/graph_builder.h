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
#ifndef INTELLGRAPH_SRC_GRAPH_GRAPH_BUILDER_H_
#define INTELLGRAPH_SRC_GRAPH_GRAPH_BUILDER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "src/graph/classifier_impl.h"
#include "src/proto/edge_parameter.pb.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/proto/vertex_parameter.pb.h"

namespace intellgraph {

template <typename T> class GraphBuilder {
public:
  GraphBuilder();
  ~GraphBuilder();

  GraphBuilder<T> &add_edge(int edge_id, const std::string &edge_type,
                            const VertexParameter &vtx_param_in,
                            const VertexParameter &vtx_param_out);
  GraphBuilder<T> &add_vertex(const VertexParameter &vtx_param);
  GraphBuilder<T> &add_edge(const EdgeParameter &edge_param);
  GraphBuilder<T> &add_solver(const SolverConfig &solver_config);
  GraphBuilder<T> &set_length(int length);
  const GraphParameter &graph_parameter();
  ClassifierImpl<T> BuildClassifier();

private:
  int length_ = 0;
  std::set<int> vertex_ids_;
  std::set<int> edge_ids_;
  std::set<std::pair<int, int>> edges_;
  GraphParameter graph_parameter_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class GraphBuilder<float>;
extern template class GraphBuilder<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_GRAPH_BUILDER_H_
