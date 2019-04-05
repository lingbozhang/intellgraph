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

#include <string>
#include <unordered_map>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/topological_sort.hpp"
#include "edge/dense_edge.h"
#include "edge/edge.h"
#include "edge/edge_factory.h"
#include "edge/edge_parameter.h"
#include "node/input_node.h"
#include "node/node.h"
#include "node/node_edge_interface.h"
#include "node/node_factory.h"
#include "node/node_parameter.h"
#include "node/output_node.h"
#include "node/sigmoid_input_node.h"
#include "node/sigmoid_l2_node.h"
#include "node/sigmoid_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {

struct EdgeProperty {
  size_t id;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, \
    boost::no_property, EdgeProperty> IntellGraph;
typedef IntellGraph::vertex_descriptor VertexD;
typedef IntellGraph::edge_descriptor EdgeD;

template <class T>
class GraphEngine {
 public: 
  GraphEngine() noexcept {};

  ~GraphEngine() noexcept = default;

  // Add an edge in the neural graph. NodeParameters and EdgeParameters are 
  // stored in the GraphEngine and they will be later used to instantiate a 
  // graph object.
  void AddEdge(REF const NodeParameter<T>& node_param_in, \
               REF const NodeParameter<T>& node_param_out, \
               REF const std::string& edge_name);
  
  // Build node and edge objects based on the graph and parameters
  void Instantiate();
  //
  void Forward(MUTE MatXXSPtr<T> train_data_ptr);

  void Backward(MUTE MatXXSPtr<T> train_label_ptr, \
                COPY float learning_rate);

  void Learn(MUTE MatXXSPtr<T> train_data_ptr, \
             MUTE MatXXSPtr<T> train_label_ptr, \
             COPY float learning_rate);

  REF inline const std::vector<size_t>& ref_order() const {
    return order_;
  }
  
  inline void ClearGraph() {
    output_node_id_ = 0;
    input_node_id_ = 0;

    node_param_map_.clear();
    edge_param_map_.clear();

    output_node_ptr_ = nullptr;
    input_node_ptr_ = nullptr;

    node_map_.clear();
    edge_map_.clear();

    order_.clear();
  }

  inline void set_output_node_id(COPY const size_t id) {
    if (node_param_map_.count(id) == 0) {
      std::cout << "WARNING: node: " << id << " does not exist in the graph" 
                << std::endl;
      return;
    }
    output_node_id_ = id;
  }

  inline void set_input_node_id(COPY const size_t id) {
    if (node_param_map_.count(id) == 0) {
      std::cout << "WARNING: node: " << id << " does not exist in the graph" 
                << std::endl;
      return;
    }
    input_node_id_ = id;
  }

  REF inline const Node<T>* ref_output_node_ptr() const {
    return output_node_ptr_;
  }
 //private:
  IntellGraph graph_{};

  size_t output_node_id_{0};
  size_t input_node_id_{0};

  std::unordered_map<size_t, NodeParameter<T>> node_param_map_{};
  std::unordered_map<size_t, EdgeParameter> edge_param_map_{};

  OutputNode<T>* output_node_ptr_{nullptr};
  InputNode<T>* input_node_ptr_{nullptr};

  std::unordered_map<size_t, NodeUPtr<T>> node_map_{};
  std::unordered_map<size_t, EdgeUPtr<T>> edge_map_{};

  // Topological sorting result
  std::vector<size_t> order_{};

};

}  // intellgraph

#endif  // INTELLGRAPH_ENGINE_GRAPH_ENGINE_H_