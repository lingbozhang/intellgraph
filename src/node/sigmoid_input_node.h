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
#ifndef INTELLGRAPH_LAYER_SIGMOID_INPUT_NODE_H_
#define INTELLGRAPH_LAYER_SIGMOID_INPUT_NODE_H_

#include <functional>
#include <memory>
#include <vector>

#include "node/input_node.h"
#include "node/node_parameter.h"
#include "node/sigmoid_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// SigInputNode improves performance of CallActFxn and CalcActPrime with Eigen 
// library and has better performance than ActivationNode. 
template <class T>
class SigInputNode : implements InputNode<T> {
 public:
  explicit SigInputNode(REF const NodeParameter<T>& node_param) {
    NodeParameter<T> node_param_new;
    node_param_new.Clone(node_param);
    node_param_new.move_node_name("SigmoidNode");
    node_ptr_ = std::make_unique<SigmoidNode<T>>(node_param_new);
  }
  
  // Move constructor   
  SigInputNode(MOVE SigInputNode<T>&& rhs) noexcept = default;

  // Move operator
  SigInputNode<T>& operator=(MOVE SigInputNode<T>&& rhs) noexcept = default;
  
  // Copy constructor and operator are deleted
  SigInputNode(REF const SigInputNode<T>& rhs) = delete;
  SigInputNode<T>& operator=(REF const SigInputNode<T>& rhs) = delete;

  ~SigInputNode() noexcept final = default;

  void CalcActPrime() final {
    node_ptr_->CalcActPrime();
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

  void CallActFxn() final {
    node_ptr_->CallActFxn();
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

  MUTE virtual inline Node<T>* get_node_ptr() const {
    return node_ptr_.get();
  }

  void FeedFeature(MUTE MatXXSPtr<T> feature_ptr) final {
    this->get_activation_ptr()->array() = feature_ptr->array();
  }

 private:
  NodeUPtr<T> node_ptr_;

};

// Alias for unqiue SigInputNode pointer
template <class T>
using SigInputNodeUPtr = std::unique_ptr<SigInputNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_LAYER_SIGMOID_INPUT_NODE_H_







