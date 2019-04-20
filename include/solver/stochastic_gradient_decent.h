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
#ifndef INTELLGRAPH_SOLVER_STOCHASTIC_GRADIENT_DECENT_H_
#define INTELLGRAPH_SOLVER_STOCHASTIC_GRADIENT_DECENT_H_

#include "glog/logging.h"
#include "graph/graphfxn.h"
#include "utility/common.h"

namespace intellgraph {

template <class T>
class SGDSolver {
 public:
  SGDSolver(COPY T learning_rate, COPY T momentum_coeff = 0, COPY T lambda = 0) {
    eta_ = learning_rate;
    momentum_coeff_ = momentum_coeff;
    lambda_ = lambda;
  }

  ~SGDSolver() noexcept = default;

  void move_graph_ptr(MOVE GraphfxnUPtr<T> graph_ptr) {
    graph_ptr_ = std::move(graph_ptr);
  }

  void Train(REF const Eigen::Ref<const MatXX<T>>& training_data, \
             REF const Eigen::Ref<const MatXX<T>>& training_labels) {
    if (graph_ptr_ == nullptr) {
      LOG(ERROR) << "graph_ptr_ is not defined in SGDSolcer.";
      exit(1);
    }
    
    graph_ptr_->Derivative(training_data, training_labels);

    // Updates weights and bias
    std::vector<size_t> order = graph_ptr_->get_order();
    size_t node_out_id = graph_ptr_->get_output_node_id();
  
    for (auto it_r = order.rbegin() + 1; it_r != order.rend(); ++it_r) {
      // Stochastic gradient decent
      size_t node_in_id = *it_r;
      graph_ptr_->get_edge_weight_ptr(node_in_id, node_out_id)->array() -= \
          eta_ * graph_ptr_->get_edge_nabla_ptr(node_in_id, node_out_id)->array();
      graph_ptr_->get_node_bias_ptr(node_out_id)->array() -= eta_ * \
          graph_ptr_->get_node_delta_ptr(node_out_id)->array();

      node_out_id = node_in_id;
    }
  }

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& test_data, \
                REF const Eigen::Ref<const MatXX<T>>& test_labels) {
    graph_ptr_->Evaluate(test_data, test_labels);
  }

 private:
  T eta_{0};
  T momentum_coeff_{0};
  T lambda_{0};
  GraphfxnUPtr<T>* graph_ptr_{nullptr};

};

}  // intellgraph

#endif  // INTELLGRAPH_SOLVER_STOCHASTIC_GRADIENT_DECENT_H_