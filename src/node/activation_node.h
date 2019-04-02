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
#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/common.h"

namespace intellgraph {
// ActivationNode allows user provided function pointers. The constructor 
// accepts three parameters:
// 1. node_param: node parameter
// 2. act_function_ptr: activation function pointer
// 3. act_prime_ptr: activation prime function pointer
template <class T>
class ActivationNode : public Node<T> {
 public:
  ActivationNode() noexcept = default;

  explicit ActivationNode(const NodeParameter<T>& node_param);

  ~ActivationNode() noexcept = default;

  // Move constructor
  ActivationNode(ActivationNode<T>&& rhs) noexcept = default;

  // Move operator
  ActivationNode& operator=(ActivationNode&& rhs) noexcept = default;
  
  // Copy constructor and operator are explicitly deleted
  ActivationNode(const ActivationNode<T>& rhs) = delete;
  ActivationNode& operator=(const ActivationNode<T>& rhs) = delete; 

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;

  // Calls activation function and updates activation. Note this function calls 
  // activation function at runtime and thus has performance penalty
  void CallActFxn() final;

  // Calculates derivative of the activation function and overwrites the 
  // activation in-place. Note this function calls activation prime function at 
  // runtime and thus has performance penalty
  void CalcActPrime() final;

  void ApplyUnaryFunctor_k(const std::function<T(T)>& functor) final;

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

  inline MatXX<T>* get_c_delta_ptr() const final {
    return delta_ptr_.get();
  }

  inline void set_m_delta_ptr(MatXXUPtr<T> delta_ptr) final {
    delta_ptr_ = std::move(delta_ptr);
  }

  inline MatXX<T>* get_c_bias_ptr() const final {
    return bias_ptr_.get();
  }

  inline void set_m_bias_ptr(MatXXUPtr<T> bias_ptr) final {
    bias_ptr_ = std::move(bias_ptr);
  }

 private:
  void ActToPrime();

  void InitToAct();

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

template <class T>
using ActNodeUPtr = std::unique_ptr<ActivationNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_ACTIVATION_NODE_H_







  