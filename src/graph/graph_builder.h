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

#include "boost/graph/adjacency_list.hpp"
#include "src/graph.h"
#include "src/proto/edge_parameter.pb.h"
#include "src/proto/vertex_parameter.pb.h"
#include "src/solver.h"
#include "src/visitor.h"

namespace intellgraph {

template <typename T> class GraphBuilder {
public:
  GraphBuilder();
  ~GraphBuilder();

  void add_edge(const std::string &edge_type,
                const VertexParameter &vtx_param_in,
                const VertexParameter &vtx_param_out);
  const std::set<VertexParameter> &vertex_params();
  const std::set<EdgeParameter> &edge_params();
  const Graph::AdjacencyList &adjacency_list();

  void set_input_vertex_id(int id);
  int input_vertex_id();

  void set_output_vertex_id(int id);
  int output_vertex_id();

private:
  int input_vertex_id_ = -1;
  int output_vertex_id_ = -1;

  Graph::AdjacencyList adjacency_list_;
  std::set<VertexParameter> vertex_params_;
  std::set<EdgeParameter> edge_params_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class GraphBuilder<float>;
extern template class GraphBuilder<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_GRAPH_BUILDER_H_
