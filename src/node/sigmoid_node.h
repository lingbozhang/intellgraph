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
#ifndef INTELLGRAPH_LAYER_SIGMOID_NODE_H_
#define INTELLGRAPH_LAYER_SIGMOID_NODE_H_

#include <functional>
#include <memory>
#include <vector>

#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// SigmoidNode improves performance of CallActFxn and CalcActPrime with Eigen 
// library and has better performance than ActivationNode. 
template <class T>
class SigmoidNode : implements Node<T> {
 public:
  SigmoidNode() noexcept = default;

  explicit SigmoidNode(REF const NodeParameter<T>& node_param);

  // Move constructor
  SigmoidNode(MOVE SigmoidNode<T>&& rhs) noexcept = default;

  // Move operator
  SigmoidNode& operator=(MOVE SigmoidNode<T>&& rhs) noexcept = default;
  
  // Copy constructor and operator are explicitly deleted
  SigmoidNode(REF const SigmoidNode<T>& rhs) = delete;
  SigmoidNode& operator=(REF const SigmoidNode<T>& rhs) = delete;

  ~SigmoidNode() noexcept final = default;

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;

  void CallActFxn() final;

  void CalcActPrime() final;

  void InitializeAct(REF const std::function<T(T)>& functor) final;

  void InitializeBias(REF const std::function<T(T)>& functor) final;

  COPY inline std::vector<size_t> get_dims() const final {
    return node_param_.ref_dims();
  }

  REF inline const std::vector<size_t>& ref_dims() const final {
    return node_param_.ref_dims();
  }

  MUTE inline MatXX<T>* get_activation_ptr() const final {
    return activation_ptr_.get();
  }

  inline void move_activation_ptr(MOVE MatXXUPtr<T> activation_ptr) final {
    activation_ptr_ = std::move(activation_ptr);
    Transition(kInit);
  };

  inline void set_activation(COPY T value) final {
    activation_ptr_->array() = value;
    Transition(kInit);
  }

  MUTE inline MatXX<T>* get_bias_ptr() const final {
    return bias_ptr_.get();
  }

  inline void move_bias_ptr(MOVE MatXXUPtr<T> bias_ptr) final {
    bias_ptr_ = std::move(bias_ptr);
  }

  MUTE inline MatXX<T>* get_delta_ptr() const final {
    return delta_ptr_.get();
  }

  inline void move_delta_ptr(MOVE MatXXUPtr<T> delta_ptr) final {
    delta_ptr_ = std::move(delta_ptr);
  }

  REF inline const NodeParameter<T>& ref_node_param() const final {
    return node_param_;
  }

  // Transitions from kAct state to kPrime state and updates current_act_state_
  void ActToPrime() final;
  // Transitions from kInit state to kAct state and updates current_act_state_
  void InitToAct() final;
  // Transitions from current_act_state_ to state
  bool Transition(ActStates state) final;

 private:
  NodeParameter<T> node_param_{};
  MatXXUPtr<T> activation_ptr_{nullptr};
  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXUPtr<T> delta_ptr_{nullptr};
  MatXXUPtr<T> bias_ptr_{nullptr};
  // Stores current state of activation vector
  ActStates current_act_state_{kInit};

};

// Alias for unqiue SigmoidNode pointer
template <class T>
using SigNodeUPtr = std::unique_ptr<SigmoidNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_LAYER_SIGMOID_NODE_H_







