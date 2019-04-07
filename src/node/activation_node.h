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
#ifndef INTELLGRAPH_NODE_ACTIVATION_NODE_H_
#define INTELLGRAPH_NODE_ACTIVATION_NODE_H_

#include <functional>
#include <memory>
#include <vector>
// Your project's .h files
#include "glog/logging.h"
#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {
// ActivationNode allows user provided function pointers. ActivationNode has
// two functors;
// 1. act_function_ptr: activation function pointer
// 2. act_prime_ptr: activation prime function pointer
template <class T>
class ActivationNode : public Node<T> {
 public:
  ActivationNode() noexcept = default;

  explicit ActivationNode(REF const NodeParameter<T>& node_param);

  ~ActivationNode() noexcept final = default;

  // Move constructor
  ActivationNode(MOVE ActivationNode<T>&& rhs) noexcept = default;

  // Move operator
  ActivationNode& operator=(MOVE ActivationNode&& rhs) noexcept = default;
  
  // Copy constructor and operator are explicitly deleted
  ActivationNode(REF const ActivationNode<T>& rhs) = delete;
  ActivationNode& operator=(REF const ActivationNode<T>& rhs) = delete;

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;

  // Calls activation function and updates activation. Note this function calls 
  // activation function at runtime and thus has performance penalty
  bool CallActFxn() final;

  // Calculates derivative of the activation function and overwrites the 
  // activation in-place. Note this function calls activation prime function at 
  // runtime and thus has performance penalty
  bool CalcActPrime() final;

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
    CHECK_EQ(activation_ptr_->size(), activation_ptr->size())
        << "move_activation_ptr() for ActivationNode is failed"
        << "activation dimensions are not equal";
    activation_ptr_ = std::move(activation_ptr);
    Transition(kInit);
  };

  inline void set_activation(COPY T value) final {
    activation_ptr_->array() = value;
  }

  MUTE inline MatXX<T>* get_delta_ptr() const final {
    return delta_ptr_.get();
  }

  inline void move_delta_ptr(MOVE MatXXUPtr<T> delta_ptr) final {
    CHECK_EQ(delta_ptr_->size(), delta_ptr->size())
        << "move_delta_ptr() for ActivationNode is failed"
        << "delta dimensions are not equal";
    delta_ptr_ = std::move(delta_ptr);
  }

  MUTE inline MatXX<T>* get_bias_ptr() const final {
    return bias_ptr_.get();
  }

  inline void move_bias_ptr(MOVE MatXXUPtr<T> bias_ptr) final {
    CHECK_EQ(bias_ptr_->size(), bias_ptr->size())
        << "move_bias_ptr() for ActivationNode is failed"
        << "bias dimensions are not equal";
    bias_ptr_ = std::move(bias_ptr);
  }

  REF inline const NodeParameter<T>& ref_node_param() const final {
    return node_param_;
  }

  void ActToPrime() final;

  void InitToAct() final;

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

template <class T>
using ActNodeUPtr = std::unique_ptr<ActivationNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_ACTIVATION_NODE_H_







  