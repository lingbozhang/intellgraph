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
#ifndef INTELLGRAPH_ENGINE_GRAPH_ENGINE_H_
#define INTELLGRAPH_ENGINE_GRAPH_ENGINE_H_

#include <unordered_map>

#include "boost/graph/adjacency_list.hpp"
#include "edge/edge_factory.h"
#include "node/node_factory.h"
#include "utility/common.h"

namespace intellgraph {

struct EdgeProperty {
  size_t id;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, \
    boost::no_property, EdgeProperty> IntellGraph;
typedef IntellGraph::vertex_descriptor VertexD;
typedef IntellGraph::edge_descriptor EdgeD;

template <class T>
class GraphEngine {
 public: 
  GraphEngine() {}

  ~GraphEngine() {}
  
  void AddEdge(NodeParameter node_param_in, \
               NodeParameter node_param_out, \
               std::string edge_name) {
    // Construct node objects and put them in the node_map_
    size_t vertex_in_id = node_param_in.id;
    size_t vertex_out_id = node_param_out.id;

    EdgeParameter<T> edge_param;
    EdgeProperty edge_property;

    if (node_param_map_.count(vertex_in_id) == 0) {
      node_param_map_[vertex_in_id] = node_param_in;
    }

    if (node_param_map_.count(vertex_out_id) == 0) {
      node_param_map_[vertex_out_id] = node_param_out;
    }
    edge_property.id = edge_param_map_.size();

    if (edge_param_map_.count(edge_property.id) > 0) {
      std::cout << "WARNING: Edge " << edge_property.id 
                << "has been added, and is skipped" << std::endl;
    } else {
      edge_param.id = edge_property.id;
      edge_param.edge_name = edge_name;
      edge_param.dims_in = node_param_in.dims;
      edge_param.dims_out = node_param_out.dims;

      boost::add_edge(vertex_in_id, vertex_out_id, edge_property, graph_);
      edge_param_map_[edge_param.id] = edge_param;
    }
  }
//private:
  IntellGraph graph_;

  std::unordered_map<size_t, NodeParameter> node_param_map_;
  std::unordered_map<size_t, EdgeParameter<T>> edge_param_map_;
};

}  // intellgraph

#endif  // INTELLGRAPH_ENGINE_GRAPH_ENGINE_H_