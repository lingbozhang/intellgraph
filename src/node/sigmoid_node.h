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
#include <vector>

#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/common.h"

namespace intellgraph {
// SigmoidNode improves performance of CallActFxn and CalcActPrime with Eigen 
// library and has better performance than ActivationNode. 
template <class T>
class SigmoidNode : public Node<T> {
 public:
  explicit SigmoidNode(const NodeParameter& node_param);

  ~SigmoidNode() {}

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;

  void CallActFxn() final;

  void CalcActPrime() final;

  void ApplyUnaryFunctor(std::function<T(T)> functor) final;

  inline std::vector<size_t> GetDims() final {
    return node_param_.dims;
  }

  inline MatXXSPtr<T> GetActivationPtr() final {
    return activation_ptr_;
  }

  inline void SetActivationPtr(MatXXSPtr<T>& activation_ptr) final {
    activation_ptr_ = activation_ptr;
    Transition(kInit);
  };

  inline void SetActivation(T value) final {
    activation_ptr_->array() = value;
    Transition(kInit);
  }

  inline MatXXSPtr<T> GetBiasPtr() final {
    return bias_ptr_;
  }

  inline void SetBiasPtr(MatXXSPtr<T>& bias_ptr) final {
    bias_ptr_ = bias_ptr;
  }

  inline MatXXSPtr<T> GetDeltaPtr() final {
    return delta_ptr_;
  }

  inline void SetDeltaPtr(MatXXSPtr<T>& delta_ptr) final {
    delta_ptr_ = delta_ptr;
  }

  inline bool IsActivated() final {
    return current_act_state_ == kAct;
  }

 private:
  // Transitions from kAct state to kPrime state and updates current_act_state_
  void ActToPrime();
  // Transitions from kInit state to kAct state and updates current_act_state_
  void InitToAct();
  // Transitions from current_act_state_ to state
  bool Transition(ActStates state);

  const NodeParameter node_param_;
  MatXXSPtr<T> activation_ptr_;
  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXSPtr<T> delta_ptr_;
  MatXXSPtr<T> bias_ptr_;
  // Stores current state of activation vector
  ActStates current_act_state_;
};

// Alias for shared SigmoidNode pointer
template <class T>
using SigNodeSPtr = std::shared_ptr<SigmoidNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_LAYER_SIGMOID_NODE_H_







