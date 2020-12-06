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
#include "src/graph/graph_builder.h"

#include "glog/logging.h"
#include "src/proto/vertex_parameter.pb.h"

namespace intellgraph {

static bool operator<(const VertexParameter &left,
                      const VertexParameter &right) {
  return left.id() < right.id();
}

template <typename T> GraphBuilder<T>::GraphBuilder() = default;
template <typename T> GraphBuilder<T>::~GraphBuilder() = default;

template <typename T>
GraphBuilder<T> &
GraphBuilder<T>::AddEdge(int edge_id, const std::string &edge_type,
                         const VertexParameter &vtx_param_in,
                         const VertexParameter &vtx_param_out) {
  DCHECK_GE(edge_id, 0);
  DCHECK(!edge_type.empty());
  DCHECK_NE(vtx_param_in.id(), vtx_param_out.id());

  // Adds vertices
  AddVertex(vtx_param_in);
  AddVertex(vtx_param_out);

  // Constructs the edge parameter
  EdgeParameter edge_param;
  edge_param.set_id(edge_id);
  edge_param.set_type(edge_type);
  edge_param.set_vertex_in_id(vtx_param_in.id());
  edge_param.set_vertex_out_id(vtx_param_out.id());
  AddEdge(edge_param);

  return *this;
}

template <typename T>
GraphBuilder<T> &GraphBuilder<T>::AddVertex(const VertexParameter &vtx_param) {
  if (vertex_ids_.count(vtx_param.id())) {
    LOG(WARNING) << "Add vertex: << " << vtx_param.id()
                 << " failed, it has already been added!";
    return *this;
  }
  switch (vtx_param.type()) {
  case VertexParameter_Type_INPUT:
    graph_parameter_.mutable_input_vertex_param()->CopyFrom(vtx_param);
    break;
  case VertexParameter_Type_HIDDEN:
    graph_parameter_.add_intermediate_vertex_params()->CopyFrom(vtx_param);
    break;
  case VertexParameter_Type_OUTPUT:
    graph_parameter_.mutable_output_vertex_param()->CopyFrom(vtx_param);
    break;
  case VertexParameter_Type_VertexParameter_Type_INT_MIN_SENTINEL_DO_NOT_USE_:
  case VertexParameter_Type_VertexParameter_Type_INT_MAX_SENTINEL_DO_NOT_USE_:
    LOG(DFATAL) << "Vertex in has an invalid type!";
    return *this;
  }
  vertex_ids_.insert(vtx_param.id());
  return *this;
}

template <typename T>
GraphBuilder<T> &GraphBuilder<T>::AddEdge(const EdgeParameter &edge_param) {
  if (!IsValidEdgeParameter(edge_param)) {
    LOG(ERROR) << "Add edge failed: not a valid edge parameter!";
    return *this;
  }
  graph_parameter_.add_edge_params()->CopyFrom(edge_param);
  edge_ids_.insert(edge_param.id());
  edges_.insert(
      std::make_pair(edge_param.vertex_in_id(), edge_param.vertex_out_id()));
  return *this;
}

template <typename T>
GraphBuilder<T> &GraphBuilder<T>::AddSolver(const SolverConfig &solver_config) {
  DCHECK_GT(solver_config.eta(), 0);
  DCHECK_GE(solver_config.lambda(), 0);

  graph_parameter_.mutable_solver_config()->CopyFrom(solver_config);
  return *this;
}

template <typename T> GraphBuilder<T> &GraphBuilder<T>::SetLength(int length) {
  DCHECK_GT(length, 0);
  length_ = length;
  return *this;
}

template <typename T> const GraphParameter &GraphBuilder<T>::graph_parameter() {
  return graph_parameter_;
}

template <typename T> ClassifierImpl<T> GraphBuilder<T>::BuildClassifier() {
  if (graph_parameter_.length() == 0) {
    LOG(ERROR)
        << "Build the Classifier failed, graph length equals 0, set default "
           "value to 1!";
    graph_parameter_.set_length(1);
  }
  LOG_IF(ERROR, !graph_parameter_.has_input_vertex_param())
      << "Build the Classifier failed, graph input vertex parameter "
         "hasn't been set!";
  LOG_IF(ERROR, !graph_parameter_.has_output_vertex_param())
      << "Build the Classifier failed, graph output vertex parameter "
         "hasn't been set!";
  LOG_IF(ERROR, !graph_parameter_.edge_params().size())
      << "Build the Classifier failed, graph edge parameters"
         "haven't been set!";
  return ClassifierImpl<T>(graph_parameter_);
}

template <typename T>
bool GraphBuilder<T>::IsValidEdgeParameter(const EdgeParameter &edge_param) {
  if (edge_ids_.count(edge_param.id())) {
    LOG(ERROR) << "Add edge: " << edge_param.id()
               << " failed, edge id has already been added in the graph!";
    return false;
  }
  if (!vertex_ids_.count(edge_param.vertex_in_id()) ||
      !vertex_ids_.count(edge_param.vertex_out_id())) {
    LOG(ERROR) << "Add edge: " << edge_param.id()
               << " failed, edge vertices haven't been added in the graph!";
    return false;
  }
  if (edges_.count(std::make_pair(edge_param.vertex_in_id(),
                                  edge_param.vertex_out_id()))) {
    LOG(ERROR) << "Add edge: " << edge_param.id()
               << " failed, edge has already been added in the graph!";
    return false;
  }
  return true;
}

// Explicit instantiation
template class GraphBuilder<float>;
template class GraphBuilder<double>;

} // namespace intellgraph
