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
#ifndef INTELLGRAPH_NODE_SIGMOID_L2_NODE_H
#define INTELLGRAPH_NODE_SIGMOID_L2_NODE_H

#include <functional>
#include <memory>
#include <vector>

#include "node/output_node.h"
#include "node/node_parameter.h"
#include "utility/common.h"

namespace intellgraph {
// SigSqrNode improves performance of GetLoss, CalcDelta, GetLoss, and 
// CalcDelta with Eigen library and has better performance than ActLossNode.
// In SigSqrNode, sigmoid function is used as a activation function and squared
// Euclidean norm is used as a loss function
template<class T>
class SigL2Node : public OutputNode<T> {
 public:
  SigL2Node() noexcept = default;

  explicit SigL2Node(const NodeParameter<T>& node_param);

  // Move constructor
  SigL2Node(SigL2Node<T>&& rhs) noexcept = default;

  // Move operator
  SigL2Node& operator=(SigL2Node<T>&& rhs) noexcept = default;

  // Copy constructor and operator are explicitly deleted
  SigL2Node(const SigL2Node<T>& rhs) = delete;
  SigL2Node& operator=(const SigL2Node<T>& rhs) = delete;

  ~SigL2Node() noexcept = default;

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;
  
  void CallActFxn() final;

  void CalcActPrime() final;

  void ApplyUnaryFunctor_k(const std::function<T(T)>& functor) final;

  T CalcLoss_k(const MatXX<T>& data_result) final;

  void CalcDelta_k(const MatXX<T>& data_result) final;

  inline std::vector<size_t> get_c_dims() const final {
    return node_param_.get_k_dims();
  }

  inline const std::vector<size_t>& get_k_dims() const final {
    return node_param_.get_k_dims();
  }

  inline MatXX<T>* get_c_activation_ptr() const final {
    return activation_ptr_.get();
  }

  inline void set_m_activation_ptr(MatXXUPtr<T> activation_ptr) final {
    activation_ptr_ = std::move(activation_ptr);
    Transition(kInit);
  };

  inline void set_c_activation(T value) final {
    activation_ptr_->array() = value;
    Transition(kInit);
  }

  inline MatXX<T>* get_c_bias_ptr() const final {
    return bias_ptr_.get();
  }

  inline void set_m_bias_ptr(MatXXUPtr<T> bias_ptr) final {
    bias_ptr_ = std::move(bias_ptr);
  }

  inline MatXX<T>* get_c_delta_ptr() const final {
    return delta_ptr_.get();
  }

  inline void set_m_delta_ptr(MatXXUPtr<T> delta_ptr) final {
    delta_ptr_ = std::move(delta_ptr);
  }

 private:
  // Transitions from kAct state to kPrime state and updates current_act_state_
  void ActToPrime();
  // Transitions from kInit state to kAct state and updates current_act_state_
  void InitToAct();
  // Transitions from current_act_state_ to state
  bool Transition(ActStates state);

  NodeParameter<T> node_param_{};
  MatXXUPtr<T> activation_ptr_{nullptr};
  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXUPtr<T> delta_ptr_{nullptr};
  MatXXUPtr<T> bias_ptr_{nullptr};
  // Stores current state of activation vector
  ActStates current_act_state_{kInit};

};
// Alias for shared SigL2Node pointer
template <class T>
using SigL2NodeUPtr = std::unique_ptr<SigL2Node<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_SIGMOID_L2_NODE_H
