/* Copyright 2019 The IntellGraph Authors. All Rights Reserved.
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
#ifndef INTELLGRAPH_GRAPH_GRPAPH_H_
#define INTELLGRAPH_GRAPH_GRPAPH_H_

#include "edge/edge_headers.h"
#include "node/node_headers.h"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/topological_sort.hpp"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

struct EdgeProperty {
  size_t id;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, \
    boost::no_property, EdgeProperty> IntellGraph;
typedef IntellGraph::vertex_descriptor VertexD;
typedef IntellGraph::edge_descriptor EdgeD;

// An interface for neural graphs
template <class T>
interface Graph {
 public:
  virtual ~Graph() noexcept = default;

  virtual void AddEdge(REF const NodeParameter& node_param_in, \
                       REF const NodeParameter& node_param_out, \
                       REF const std::string& edge_name) = 0;

  virtual void Instantiate() = 0;

  virtual void Reset() = 0;

  virtual void Forward(REF const Eigen::Ref<const MatXX<T>>& training_data, \
      REF const Eigen::Ref<const MatXX<T>>& training_labels) = 0;

  virtual void Backward(REF const Eigen::Ref<const MatXX<T>>& training_data, \
      REF const Eigen::Ref<const MatXX<T>>& training_labels) = 0;

  virtual void Evaluate(REF const Eigen::Ref<const MatXX<T>>& test_data, \
      REF const Eigen::Ref<const MatXX<T>>& test_labels) = 0;

  MUTE virtual inline MatXX<T>* get_edge_weight_ptr(COPY size_t node_in_id, \
      COPY size_t node_out_id) = 0;

  REF virtual inline const MatXX<T>* get_edge_nabla_ptr(COPY size_t node_in_id, \
      COPY size_t node_out_id) = 0;

  MUTE virtual inline VecX<T>* get_node_bias_ptr(COPY size_t node_id) = 0;

  REF virtual inline const MatXX<T>* get_node_delta_ptr(COPY size_t node_id) = 0;

  virtual inline bool set_output_node_id(COPY size_t id) = 0;

  virtual inline bool set_input_node_id(COPY size_t id) = 0;

};

}  // intellgraph

#endif  // INTELLGRAPH_GRAPH_GRPAPH_H_