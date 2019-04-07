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
#ifndef INTELLGRAPH_NODE_ACT_LOSS_NODE_H_
#define INTELLGRAPH_NODE_ACT_LOSS_NODE_H_

#include <functional>
#include <vector>
// Your project's .h files
#include "glog/logging.h"
#include "node/activation_node.h"
#include "node/output_node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// ActLossNode allows user provided function pointers. ActLossNode 
// has four functors: 
// 1. act_function_ptr: activation function pointer
// 2. act_prime_ptr: activation prime function pointer
// 3. loss_function_ptr: loss function pointer
// 4. loss_prime_ptr: loss function prime pointer
template <class T>
class ActLossNode : implements OutputNode<T> {
 public:
  ActLossNode() noexcept = default;

  explicit ActLossNode(REF const NodeParameter<T>& node_param); 

  ActLossNode(MOVE ActLossNode<T>&& rhs) noexcept = default;

  REF ActLossNode& operator=(MOVE ActLossNode<T>&& rhs) noexcept = default;

  ActLossNode(REF const ActLossNode<T>& rhs) = delete;

  REF ActLossNode& operator=(REF const ActLossNode<T>& rhs) = delete;

  ~ActLossNode() noexcept final = default;

  COPY T CalcLoss(REF const MatXX<T>* data_result_ptr) final;

  bool CalcDelta(REF const MatXX<T>* data_result_ptr) final;

  bool CalcActPrime() final {
    return node_ptr_->CalcActPrime();
  }

  MUTE inline MatXX<T>* get_activation_ptr() const final {
    return node_ptr_->get_activation_ptr();
  }

  inline void move_activation_ptr(MOVE MatXXUPtr<T> activation_ptr) final {
    node_ptr_->move_activation_ptr(std::move(activation_ptr));
  }

  inline void set_activation(COPY T value) final {
    node_ptr_->set_activation(value);
  }

  MUTE inline MatXX<T>* get_bias_ptr() const final {
    return node_ptr_->get_bias_ptr();
  }

  inline void move_bias_ptr(MOVE MatXXUPtr<T> bias_ptr) final {
    node_ptr_->move_bias_ptr(std::move(bias_ptr));
  }

  MUTE inline MatXX<T>* get_delta_ptr() const final {
    return node_ptr_->get_delta_ptr();
  }

  inline void move_delta_ptr(MOVE MatXXUPtr<T> delta_ptr) final {
    node_ptr_->move_delta_ptr(std::move(delta_ptr));
  }

  void PrintAct() const final {
    node_ptr_->PrintAct();
  }

  void PrintDelta() const final {
    node_ptr_->PrintDelta();
  }

  void PrintBias() const final {
    node_ptr_->PrintBias();
  }

  bool CallActFxn() final {
    return node_ptr_->CallActFxn();
  }

  // Passes a functor and applies it on the activation matrix
  void InitializeAct(REF const std::function<T(T)>& functor) final {
    node_ptr_->InitializeAct(functor);
  }

  void InitializeBias(REF const std::function<T(T)>& functor) final {
    node_ptr_->InitializeBias(functor);
  }

  // Get layer dimensions
  COPY inline std::vector<size_t> get_dims() const final {
    return node_ptr_->get_dims();
  }

  REF inline const std::vector<size_t>& ref_dims() const final {
    return node_ptr_->ref_dims();
  }

  REF inline const NodeParameter<T>& ref_node_param() const final {
    return node_ptr_->ref_node_param();
  }
  // Transitions from kAct state to kPrime state and updates current_act_state_
  void ActToPrime() final {
    node_ptr_->ActToPrime();
  }
  // Transitions from kInit state to kAct state and updates current_act_state_
  void InitToAct() final {
    node_ptr_->InitToAct();
  }
  // Transitions from current_act_state_ to state
  bool Transition(ActStates state) final {
    return node_ptr_->Transition(state);
  }
 private:
  NodeUPtr<T> node_ptr_;
};

template <class T>
using ActLossNodeUPtr = std::unique_ptr<ActLossNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_ACT_LOSS_NODE_H_







  