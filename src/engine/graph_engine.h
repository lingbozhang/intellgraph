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

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, \
    boost::no_property, boost::no_property> IntellGraph;
typedef IntellGraph::vertex_descriptor VertexD;
typedef IntellGraph::edge_descriptor EdgeD;

template <class T>
class GraphEngine {
 public: 
  GraphEngine() {}

  ~GraphEngine() {}
  
  void AddEdge(const NodeParameter& node_param_in, \
               const NodeParameter& node_param_out, \
               std::string& edge_fxn_name) {
    NodeSPtr<T> node_in = NodeFactory<T, Node<T>>::Instantiate(node_param_in);
    NodeSPtr<T> node_out = NodeFactory<T, Node<T>>::Instantiate(node_param_out);

    VertexD vertex_d_in = node_param_in.id;
    VertexD vertex_d_out = node_param_out.id;

    boost::add_edge(vertex_d_in, vertex_d_out);

    //node_map_[]
  }
 private:
  IntellGraph graph_;
  std::unordered_map<VertexD, NodeSPtr<T>> node_map_;
  std::unordered_map<EdgeD, EdgeSPtr<T>> edge_map_;
};

}  // intellgraph

#endif  // INTELLGRAPH_ENGINE_GRAPH_ENGINE_H_