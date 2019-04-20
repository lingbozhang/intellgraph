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
#ifndef INTELLGRAPH_SOLVER_GRADIENT_DECENT_H_
#define INTELLGRAPH_SOLVER_GRADIENT_DECENT_H_

#include "glog/logging.h"
#include "graph/graphfxn.h"
#include "utility/common.h"

namespace intellgraph {

template <class T>
class GDSolver {
 public:
  GDSolver(COPY T learning_rate, COPY T lambda = 0) {
    eta_ = learning_rate;
    lambda_ = lambda;
  }

  ~GDSolver() noexcept = default;

  void Train(REF const Eigen::Ref<const MatXX<T>>& training_data, \
      REF const Eigen::Ref<const MatXX<T>>& training_labels, \
      MUTE Graphfxn<T>* graph_ptr) {
    
    if (graph_ptr == nullptr) {
      LOG(ERROR) << "graph_ptr is not defined in GDSolcer.";
      exit(1);
    }

    size_t minbatch_size = training_data.cols();

    graph_ptr->Derivative(training_data, training_labels);

    // Updates weights and bias
    std::vector<size_t> order = graph_ptr->get_order();
    size_t node_out_id = graph_ptr->get_output_node_id();
  
    for (auto it_r = order.begin() + 1; it_r != order.end(); ++it_r) {
      // Stochastic gradient decent
      size_t node_in_id = *it_r;
      graph_ptr->get_edge_weight_ptr(node_in_id, node_out_id)->array() = \
          (1.0 - eta_ * lambda_) * \
          graph_ptr->get_edge_weight_ptr(node_in_id, node_out_id)->array() - \
          eta_ * graph_ptr->get_edge_nabla_ptr(node_in_id, node_out_id)->array();
      
      graph_ptr->get_node_bias_ptr(node_out_id)->array() -= \
          eta_ / minbatch_size * \
          graph_ptr->get_node_delta_ptr(node_out_id)->rowwise().sum().array();

      node_out_id = node_in_id;
    }
  }

 private:
  T eta_{0};
  T lambda_{0};

};

}  // intellgraph

#endif  // INTELLGRAPH_SOLVER_GRADIENT_DECENT_H_