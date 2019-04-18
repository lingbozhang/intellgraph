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
#include "graph/classifier.h"

namespace intellgraph {

template <class T>
void Classifier<T>::AddEdge(const NodeParameter& node_param_in, \
                            const NodeParameter& node_param_out, \
                            const std::string& edge_name) {
  // Construct node objects and put them in the node_map_
  size_t node_in_id = node_param_in.ref_id();
  size_t node_out_id = node_param_out.ref_id();

  EdgeParameter edge_param{};
  EdgeProperty edge_property{};

  if (node_param_map_.count(node_in_id) == 0) {
    index_map_[node_in_id] = node_param_map_.size();
    node_param_map_[node_in_id].Clone(node_param_in);
  }

  if (node_param_map_.count(node_out_id) == 0) {
    index_map_[node_out_id] = node_param_map_.size();
    node_param_map_[node_out_id].Clone(node_param_out);
  }
  
  edge_property.id = edge_param_map_.size();

  // Constructs EdgeParameter
  edge_param.set_id(edge_property.id);
  edge_param.set_edge_name(edge_name);
  edge_param.set_dims_in(node_param_in.ref_dims());
  edge_param.set_dims_out(node_param_out.ref_dims());

  VertexD vtx_in_id = index_map_[node_in_id];
  VertexD vtx_out_id = index_map_[node_out_id];

  if (!boost::add_edge(vtx_in_id, vtx_out_id, edge_property, graph_). \
      second) {
    LOG(WARNING) << "Edge has already been added";
    return;
  }
  edge_param_map_[edge_property.id].Clone(edge_param);
}

template <class T>
void Classifier<T>::Instantiate() {
  LOG(INFO) << "========================="
            << "INSTANTIATING CLASSIFIER"
            << "=========================";

  node_map_.clear();
  edge_map_.clear();
  input_node_ptr_ = nullptr;
  output_node_ptr_ = nullptr;

  // Instantiates the input node object
  auto input_node_ptr = std::move(NodeFactory<T, Node<T>>::Instantiate( \
      node_param_map_[input_node_id_]));
  input_node_ptr_ = input_node_ptr.get();
  node_map_[index_map_[input_node_id_]] = std::move(input_node_ptr);
  
  // Instantiates the output node object
  auto output_node_ptr = std::move(NodeFactory<T, OutputNode<T>>::Instantiate( \
      node_param_map_[output_node_id_]));
  output_node_ptr_ = output_node_ptr.get();
  node_map_[index_map_[output_node_id_]] = std::move(output_node_ptr);

  LOG(INFO) << "Initializes output node with standard normal distribution";
  output_node_ptr_->InitializeBias(NormalFunctor<T>(0.0, 1.0));

  // Instantiates node objects;
  for (const auto& node_pair : node_param_map_) {
    size_t node_id = node_pair.first;
    if (node_id != output_node_id_ && node_id != input_node_id_) {
      NodeUPtr<T> node_ptr = \
          std::move(NodeFactory<T, Node<T>>::Instantiate(node_pair.second));
      LOG(INFO) << "Initializes node: " << node_id
                << ", with standard normal distribution";
      node_ptr->InitializeBias(NormalFunctor<T>(0.0, 1.0));
      node_map_[index_map_[node_id]] = std::move(node_ptr);
    }
  }
  // Instantiates edge objects;
  for (const auto& edge_pair : edge_param_map_) {
    EdgeUPtr<T> edge_ptr = std::move(EdgeFactory<T, Edge<T>>::Instantiate( \
        edge_pair.second));
    // Initializes weight matrix with standard normal distribution
    LOG(INFO) << "Initializes edge: " << edge_pair.first
              << ", with standard normal distribution";
    edge_ptr->InitializeWeight(NormalFunctor<T>(0.0, 1.0));
    edge_map_[edge_pair.first] = std::move(edge_ptr);
  }

  order_.clear();
  LOG(INFO) << "Determines Forward() orders by topological sorting";
  topological_sort(graph_, std::back_inserter(order_));
  instantiated_ = true;
}

template <class T>
void Classifier<T>::Forward(const Eigen::Ref<const MatXX<T>>& training_data, \
                            const Eigen::Ref<const MatXX<T>>& training_labels) {
  LOG(INFO) << "======================"
            << "FORWARDING . . ."
            << "======================";

  if (!instantiated_) {
     LOG(WARNING) << "Classifier is not instantiated: "
                  << "Try to instantiate it";
     Instantiate();
  }
  if (*order_.rbegin() != index_map_[input_node_id_ ] || \
      *order_.begin() != index_map_[output_node_id_]) {
     LOG(ERROR) << "Forward() in the Classifier is failed.";
     exit(1);
  }

  input_node_ptr_->FeedFeature(training_data);
  for (auto it_r = order_.rbegin(); it_r != order_.rend(); ++it_r) {
    IntellGraph::out_edge_iterator eo{}, eo_end{};
    size_t vtx_in_id = *it_r;
    LOG(INFO) << "Forwarding vertex: " << vtx_in_id;
    if (*it_r != index_map_[input_node_id_]) node_map_[*it_r]->ToAct();
    for (std::tie(eo, eo_end) = out_edges(*it_r, graph_); eo != eo_end; ++eo) {
      size_t vtx_out_id = target(*eo, graph_);
      size_t edge_id = graph_[*eo].id;

      Node<T> *node_in_ptr, *node_out_ptr;

      node_in_ptr = node_map_[vtx_in_id].get();
        
      node_out_ptr = node_map_[vtx_out_id].get();
 
      edge_map_[edge_id]->Forward(node_in_ptr, node_out_ptr);
    }
  }
}

template <class T>
void Classifier<T>::Backward(const Eigen::Ref<const MatXX<T>>& training_data, \
                             const Eigen::Ref<const MatXX<T>>& training_labels) {
  Forward(training_data, training_labels);
  LOG(INFO) << "======================"
            << "BACKWARDING . . ."
            << "======================";
  output_node_ptr_->CalcDelta(training_labels);
  for (auto it = order_.begin(); it != order_.end(); ++it) {
    IntellGraph::in_edge_iterator ei{}, ei_end{};
    size_t vtx_out_id = *it;
    LOG(INFO) << "Backpropagating vertex: " << vtx_out_id;
    for (std::tie(ei, ei_end) = in_edges(*it, graph_); ei != ei_end; ++ei) {
      size_t vtx_in_id = source(*ei, graph_);
      size_t edge_id = graph_[*ei].id;

      Node<T> *node_in_ptr, *node_out_ptr;
      Edge<T> *edge_ptr;

      node_in_ptr = node_map_[vtx_in_id].get();
      node_out_ptr = node_map_[vtx_out_id].get();
      edge_ptr = edge_map_[edge_id].get();

      edge_ptr->Backward(node_in_ptr, node_out_ptr);
    }
  }
}

template <class T>
void Classifier<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& test_data, \
                             const Eigen::Ref<const MatXX<T>>& test_labels) {
  if (*order_.rbegin() != index_map_[input_node_id_] || \
      *order_.begin() != index_map_[output_node_id_]) {
    LOG(ERROR) << "Evaluate() in the Classifier is failed.";
    exit(1);
  }

  LOG(INFO) << "======================"
            << "FORWARDING . . ."
            << "======================";

  input_node_ptr_->FeedFeature(test_data);
  for (auto it_r = order_.rbegin(); it_r != order_.rend(); ++it_r) {
    IntellGraph::out_edge_iterator eo{}, eo_end{};
    size_t vtx_in_id = *it_r;
    LOG(INFO) << "Forwarding vertex: " << vtx_in_id;
    if (*it_r != index_map_[input_node_id_] ) node_map_[*it_r]->ToAct();
    for (std::tie(eo, eo_end) = out_edges(*it_r, graph_); eo != eo_end; ++eo) {
      size_t vtx_out_id = target(*eo, graph_);
      size_t edge_id = graph_[*eo].id;

      Node<T> *node_in_ptr, *node_out_ptr;

      node_in_ptr = node_map_[vtx_in_id].get();

      node_out_ptr = node_map_[vtx_out_id].get();

      edge_map_[edge_id]->Forward(node_in_ptr, node_out_ptr);
    }
  }
  output_node_ptr_->Evaluate(test_labels);
}

// Instantiate class, otherwise compilation will fail
template class Classifier<float>;
template class Classifier<double>;
}  // intellgraph