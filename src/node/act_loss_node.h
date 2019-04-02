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
#include "node/output_node.h"
#include "node/node_parameter.h"
#include "utility/common.h"

namespace intellgraph {
// ActLossNode allows user provided function pointers. ActLossNode 
// constructor accepts five parameters: 
// 1. node_param: node paramters
// 2. act_function_ptr: activation function pointer
// 3. act_prime_ptr: activation prime function pointer
// 4. loss_function_ptr: loss function pointer
// 5. loss_prime_ptr: loss function prime pointer
template <class T>
class ActLossNode : public OutputNode<T> {
 public:
  ActLossNode() noexcept = default;

  explicit ActLossNode(const NodeParameter<T>& node_param);

  ActLossNode(ActLossNode<T>&& rhs) noexcept = default;

  ActLossNode& operator=(ActLossNode<T>&& rhs) noexcept = default;

  ActLossNode(const ActLossNode<T>& rhs) = delete;

  ActLossNode& operator=(const ActLossNode<T>& rhs) = delete;

  ~ActLossNode() noexcept = default;
  
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

  // Note this function calls loss function at runtime and thus has performance
  // penalty
  T CalcLoss_k(const MatXX<T>& data_result) final;

  // Calculates derivative of loss function of weighted_sum variables. Note this
  // function calls loss prime function at runtime and thus has performance
  // penalty
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
using ActLossNodeUPtr = std::unique_ptr<ActLossNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_ACT_LOSS_NODE_H_







  