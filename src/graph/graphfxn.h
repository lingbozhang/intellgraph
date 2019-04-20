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
#ifndef INTELLGRAPH_GRAPH_GRPAPHFXN_H_
#define INTELLGRAPH_GRAPH_GRPAPHFXN_H_

#include <memory>
#include <unordered_map>
#include <vector>

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

// IntellGraph implements Boost Graph library and stores node and edge
// information in the adjacency list.
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, \
    boost::no_property, EdgeProperty> IntellGraph;
typedef IntellGraph::vertex_descriptor VertexD;
typedef IntellGraph::edge_descriptor EdgeD;

template <class T>
class Graphfxn {
 public:
  Graphfxn() noexcept {}

  // Move constructor
  Graphfxn(MOVE Graphfxn<T>&& rhs) = default;

  // Move operator
  Graphfxn& operator=(MOVE Graphfxn<T>&& rhs) = default;
  
  // Copy constructor and operator are explicitly deleted
  Graphfxn(REF const Graphfxn<T>& rhs) = delete;
  Graphfxn& operator=(REF const Graphfxn<T>& rhs) = delete;

  virtual ~Graphfxn() noexcept = default;

  MUTE virtual Graphfxn& AddEdge( \
      REF const std::pair<std::string, std::vector<size_t>>& node_pair_in, \
      REF const std::pair<std::string, std::vector<size_t>>& node_pair_out, \
      REF const std::string& edge_name);

  virtual void AddEdge(REF const NodeParameter& node_param_in, \
                       REF const NodeParameter& node_param_out, \
                       REF const std::string& edge_name);

  virtual void Create();

  virtual void RemoveEdge(COPY size_t node_in_id, COPY size_t node_out_id);

  virtual void ClearGraph();

  virtual void Forward(REF const Eigen::Ref<const MatXX<T>>& training_data, \
      REF const Eigen::Ref<const MatXX<T>>& training_labels);

  virtual void Derivative(REF const Eigen::Ref<const MatXX<T>>& training_data, \
      REF const Eigen::Ref<const MatXX<T>>& training_labels);

  virtual void Evaluate(REF const Eigen::Ref<const MatXX<T>>& test_data, \
      REF const Eigen::Ref<const MatXX<T>>& test_labels);

  MUTE inline MatXX<T>* get_edge_weight_ptr(COPY size_t vtx_in, \
                                            COPY size_t vtx_out) {
    auto edge_pair = boost::edge(vtx_in, vtx_out, graph_);
    if (!edge_pair.second) {
      LOG(ERROR) << "Edge connects Vertexes: " << vtx_in << ", and "
                 << vtx_out << ", dose not exist";
      return nullptr; 
    }
    EdgeD edge_descriptor = edge_pair.first;
    size_t edge_id = graph_[edge_descriptor].id;
    if (edge_map_.count(edge_id) > 0) {
      return edge_map_[edge_id].get()->get_weight_ptr();
    } else {
      LOG(ERROR) << "Edge: " << edge_id << "does not exist.";
      return nullptr;
    }
  }

  REF inline const MatXX<T>* get_edge_nabla_ptr(COPY size_t vtx_in, \
                                                COPY size_t vtx_out) {
    auto edge_pair = boost::edge(vtx_in, vtx_out, graph_);
    if (!edge_pair.second) {
      LOG(ERROR) << "Edge connects Vertexes: " << vtx_in << ", and "
                 << vtx_out << ", does not exist";
      return nullptr; 
    }
    EdgeD edge_descriptor = edge_pair.first;
    size_t edge_id = graph_[edge_descriptor].id;
    if (edge_map_.count(edge_id) > 0) {
      return edge_map_[edge_id].get()->ref_nabla_weight_ptr();
    } else {
      LOG(ERROR) << "Edge: " << edge_id << "does not exist.";
      return nullptr;
    }
  }

  MUTE inline VecX<T>* get_node_bias_ptr(COPY size_t vtx_id) {
    if (node_map_.count(vtx_id) > 0) {
      return node_map_[vtx_id]->get_bias_ptr();
    } else {
      LOG(ERROR) << "Vertex: " << vtx_id << "does not exist.";
      return nullptr;
    }
  }

  REF inline const MatXX<T>* get_node_delta_ptr(COPY size_t vtx_id) {
    if (node_map_.count(vtx_id) > 0) {
      return node_map_[vtx_id]->get_delta_ptr();
    } else {
      LOG(ERROR) << "Vertex: " << vtx_id << "does not exist.";
      return nullptr;
    }
  }

  inline bool set_output_node_id(COPY const size_t id) {
    if (node_param_map_.count(id) == 0) {
      LOG(ERROR) << "node: " << id << " does not exist." ;
      return false;
    }
    output_node_id_ = id;
    return true;
  }

  COPY inline T get_output_node_id() {
    return output_node_id_;
  }

  inline bool set_input_node_id(COPY const size_t id) {
    if (node_param_map_.count(id) == 0) {
      LOG(ERROR) << "node: " << id << " does not exist in the graph."; 
      return false;
    }
    input_node_id_ = id;
    return true;
  }

  COPY inline T get_input_node_id() {
    return input_node_id_;
  }

  inline void TurnDropoutOn(T dropout_p) {
    dropout_on_ = true;
    CHECK_GT(1.0, dropout_p) << "TurnDropoutOn() for Node is failed.";
    dropout_p_ = dropout_p;
  }

  inline void TurnDropoutOff() {
    dropout_on_ = false;
    dropout_p_ = 1.0;
  }

  COPY inline std::vector<size_t> get_order() {
    return order_;
  }

 private:
  IntellGraph graph_{};

  bool is_created_{false};

  size_t output_node_id_{0};
  size_t input_node_id_{0};

  std::unordered_map<size_t, NodeParameter> node_param_map_{};
  std::unordered_map<size_t, EdgeParameter> edge_param_map_{};

  OutputNode<T>* output_node_ptr_{nullptr};
  Node<T>* input_node_ptr_{nullptr};

  std::unordered_map<size_t, NodeUPtr<T>> node_map_{};
  std::unordered_map<size_t, EdgeUPtr<T>> edge_map_{};

  // Topological sorting result
  std::vector<size_t> order_{};
  std::vector<bool> visited_{};

  // A dropout flag
  bool dropout_on_{false};

  T dropout_p_{1.0};

};

template <class T>
using GraphfxnUPtr = std::unique_ptr<Graphfxn<T>>;

}  // intellgraph

#endif  // INTELLGRAPH_GRAPH_GRPAPHFXN_H_