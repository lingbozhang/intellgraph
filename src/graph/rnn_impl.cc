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
#include "src/graph/rnn_impl.h"

#include "boost/graph/adjacency_list.hpp"
#include "glog/logging.h"
#include "src/factory.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/proto/vertex_parameter.pb.h"
#include "src/visitor/backward_visitor.h"
#include "src/visitor/forward_visitor.h"
#include "src/visitor/init_vertex_visitor.h"
#include "src/visitor/resize_vertex_visitor.h"

namespace intellgraph {

template <typename T>
RnnImpl<T>::RnnImpl(const GraphParameter &graph_parameter)
    : Graph<T>(graph_parameter.edge_params()),
      sequence_length_(graph_parameter.length()) {
  DCHECK_GT(sequence_length_, 0);

  if (graph_parameter.has_solver_config()) {
    solver_ =
        Factory::InstantiateSolver<Solver<T>>(graph_parameter.solver_config());
  }

  // Builds the default |threshold_|
  threshold_ = MatrixX<T>(graph_parameter.output_vertex_param().dims(), 1);
  threshold_.setConstant(0.5);

  // Instantiates the input vertex
  std::unique_ptr<InputVertex<T>> input_vertex =
      Factory::InstantiateVertex<InputVertex<T>>(
          graph_parameter.input_vertex_param(), sequence_length_);
  input_vertex_ = input_vertex.get();
  vertex_by_id_.try_emplace(graph_parameter.input_vertex_param().id(),
                            std::move(input_vertex));

  // Instantiates intermediate vertices
  for (const auto &vertex_param :
       graph_parameter.intermediate_vertex_params()) {
    std::unique_ptr<OpVertex<T>> vertex =
        Factory::InstantiateVertex<OpVertex<T>>(vertex_param, sequence_length_);
    vertex_by_id_.try_emplace(vertex_param.id(), std::move(vertex));
  }

  // Instantiates output vertices
  std::unique_ptr<OutputVertex<T>> output_vertex =
      Factory::InstantiateVertex<OutputVertex<T>>(
          graph_parameter.output_vertex_param(), sequence_length_);
  output_vertex_ = output_vertex.get();
  vertex_by_id_.try_emplace(graph_parameter.output_vertex_param().id(),
                            std::move(output_vertex));

  // Instantiates edges
  for (const auto &edge_param : graph_parameter.edge_params()) {
    int edge_id = edge_param.id();
    const std::string &edge_type = edge_param.type();
    int vtx_in_id = edge_param.vertex_in_id();
    int vtx_out_id = edge_param.vertex_out_id();

    edge_by_id_.try_emplace(
        edge_id, Factory::InstantiateEdge<Edge<T>, OpVertex<T>, OpVertex<T>>(
                     edge_type, edge_id, vertex_by_id_.at(vtx_in_id).get(),
                     vertex_by_id_.at(vtx_out_id).get()));
  }
}

} // namespace intellgraph
