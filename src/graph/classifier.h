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
#ifndef INTELLGRAPH_GRAPH_CLASSIFIER_H_
#define INTELLGRAPH_GRAPH_CLASSIFIER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "graph/graph.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {

template <class T>
class Classifier : implements Graph<T> {
 public: 
  Classifier() = default;

  ~Classifier() noexcept final = default;

  // Add an edge in the neural graph. NodeParameters and EdgeParameters are 
  // stored in the hashtables and they will be later used to instantiate a 
  // graph object.
  void AddEdge(REF const NodeParameter<T>& node_param_in, \
               REF const NodeParameter<T>& node_param_out, \
               REF const std::string& edge_name) final;

  // Build node and edge objects based on the graph and parameters
  void Instantiate() final;

  void Reset() final {
    instantiated_ = false;

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

  void Forward(MUTE MatXXSPtr<T> train_data_ptr, \
               MUTE MatXXSPtr<T> train_label_ptr) final;

  void Backward(MUTE MatXXSPtr<T> train_data_ptr, \
                MUTE MatXXSPtr<T> train_label_ptr) final;

  void Predict(MUTE MatXXSPtr<T> test_data_ptr) final;

  MUTE inline MatXX<T>* get_edge_weight(COPY size_t node_in_id, \
                                        COPY size_t node_out_id) final {
    auto edge_pair = boost::edge(node_in_id, node_out_id, graph_);
    if (!edge_pair.second) {
      LOG(ERROR) << "Edge connects Nodes: " << node_in_id << " and "
                 << node_out_id << "dose not exist";
      return nullptr; 
    }
    EdgeD edge_descriptor = edge_pair.first;
    size_t edge_id = graph_[edge_descriptor].id;
    if (edge_map_.count(edge_id) > 0) {
      return edge_map_[edge_id].get()->get_weight_ptr();
    } else {
      LOG(ERROR) << "Edge: " << edge_id << "does not exist in the table."; 
      return nullptr;
    }
  }

  REF inline const MatXX<T>* get_edge_nabla(COPY size_t node_in_id, \
                                            COPY size_t node_out_id) final {
    auto edge_pair = boost::edge(node_in_id, node_out_id, graph_);
    if (!edge_pair.second) {
      LOG(ERROR) << "Edge connects Nodes: " << node_in_id << " and "
                 << node_out_id << "dose not exist";
      return nullptr; 
    }
    EdgeD edge_descriptor = edge_pair.first;
    size_t edge_id = graph_[edge_descriptor].id;
    if (edge_map_.count(edge_id) > 0) {
      return edge_map_[edge_id].get()->ref_nabla_weight_ptr();
    } else {
      LOG(ERROR) << "Edge: " << edge_id << "does not exist in the table"; 
      return nullptr;
    }
  }

  MUTE inline MatXX<T>* get_node_bias(COPY size_t node_id) final {
    if (node_map_.count(node_id) > 0) {
      return node_map_[node_id]->get_bias_ptr();
    } else {
      LOG(ERROR) << "node: " << node_id << "does not exist.";
      return nullptr;
    }
  }

  REF inline const MatXX<T>* get_node_delta(COPY size_t node_id) final {
    if (node_map_.count(node_id) > 0) {
      return node_map_[node_id]->get_delta_ptr();
    } else {
      LOG(ERROR) << "node: " << node_id << "does not exist."; 
      return nullptr;
    }
  }

  inline bool set_output_node_id(COPY const size_t id) final {
    if (node_param_map_.count(id) == 0) {
      LOG(ERROR) << "node: " << id << " does not exist in the graph." ;
      return false;
    }
    output_node_id_ = id;
    return true;
  }

  inline bool set_input_node_id(COPY const size_t id) final {
    if (node_param_map_.count(id) == 0) {
      LOG(ERROR) << "node: " << id << " does not exist in the graph."; 
      return false;
    }
    input_node_id_ = id;
    return true;
  }

 private:
  inline void CalcLoss(MUTE MatXXSPtr<T> train_label_ptr) {
    T loss = output_node_ptr_->CalcLoss(train_label_ptr.get());
    std::cout << "Loss: " << loss << std::endl;
  }

  inline void Evaluate(MatXXSPtr<T> label_ptr) {
    MatXX<T> result = output_node_ptr_->get_activation_ptr()->matrix();
    std::cout << "Evaluated/Actual results: " << result
              << "/" << label_ptr->array() << std::endl;
  }  

  IntellGraph graph_{};

  bool instantiated_{false};

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

#endif  // INTELLGRAPH_GRAPH_CLASSIFIER_H_