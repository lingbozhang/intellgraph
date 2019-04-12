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
#ifndef INTELLGRAPH_NODE_SIGMOID_CROSS_ENTROPY_H_
#define INTELLGRAPH_NODE_SIGMOID_CROSS_ENTROPY_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "node/node_parameter.h"
#include "node/output_node.h"
#include "node/sigmoid_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// SigCENode uses decorator pattern (see https://dzone.com/articles/is-inheritance-dead)
// SigCENode improves performance of GetLoss, CalcDelta, GetLoss, and CalcDelta 
// with Eigen library. In SigCENode, Identity function is used as a activation 
// function and the cross-entropy function is used as a loss function.
template<class T>
class SigCENode : public OutputNode<T> {
 public:
  SigCENode() noexcept = delete;

  explicit SigCENode(REF const NodeParameter& node_param) {
    sigmoid_node_ptr_ = std::make_unique<SigmoidNode<T>>(node_param);
  }

  // Move constructor
  SigCENode(MOVE SigCENode<T>&& rhs) noexcept = default;

  // Move operator
  SigCENode& operator=(MOVE SigCENode<T>&& rhs) noexcept = default;

  // Copy constructor and operator are explicitly deleted
  SigCENode(REF const SigCENode<T>& rhs) = delete;
  SigCENode& operator=(REF const SigCENode<T>& rhs) = delete;

  ~SigCENode() noexcept final = default;

  COPY inline std::vector<size_t> get_dims() const final {
    return sigmoid_node_ptr_->get_dims();
  }

  REF inline const std::vector<size_t>& ref_dims() const final {
    return sigmoid_node_ptr_->ref_dims();
  }

  // Accessable operations for the activation matrix
  MUTE inline MatXX<T>* get_activation_ptr() final {
    return sigmoid_node_ptr_->get_activation_ptr();
  }

  // Accessable operations for the bias vector
  MUTE inline VecX<T>* get_bias_ptr() const final {
    return sigmoid_node_ptr_->get_bias_ptr();
  }
  // Accessable operations for the delta matrix
  MUTE inline MatXX<T>* get_delta_ptr() final {
    return sigmoid_node_ptr_->get_delta_ptr();
  }
  // Accessable operations for the node parameter
  REF inline const NodeParameter& ref_node_param() const final {
    return sigmoid_node_ptr_->ref_node_param();
  }

  inline void set_activation(COPY T value) final {
    sigmoid_node_ptr_->set_activation(value);
  }

  void InitializeBias(REF const std::function<T(T)>& functor) final {
    sigmoid_node_ptr_->InitializeBias(functor);
  }

  void PrintBias() const final {
    sigmoid_node_ptr_->PrintBias();
  }

  bool CallActFxn() final {
    return sigmoid_node_ptr_->CallActFxn();
  }

  bool CalcActPrime() final {
    return sigmoid_node_ptr_->CalcActPrime();
  }

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) {
    sigmoid_node_ptr_->Evaluate(labels);
  }

  inline bool ResetActState() final {
    return sigmoid_node_ptr_->ResetActState();
  }

  void FeedFeature(REF const Eigen::Ref<const MatXX<T>>& feature) final {
    sigmoid_node_ptr_->FeedFeature(feature);
  }

  inline void TurnDropoutOn(T dropout_p) final {
    sigmoid_node_ptr_->TurnDropoutOn(dropout_p);
  }

  inline void TurnDropoutOff() final {
    sigmoid_node_ptr_->TurnDropoutOff();
  }

  COPY T CalcLoss(REF const Eigen::Ref<const MatXX<T>>& labels) final;

  bool CalcDelta(REF const Eigen::Ref<const MatXX<T>>& labels) final;
  
 protected:
  // Transitions from kInit to kAct and updates current_act_state_
  void InitToAct() final {
    sigmoid_node_ptr_->InitToAct();
  }

  // Transitions from kAct state to kDropout state and updates current_act_state_
  void ActToDropout() final {
    sigmoid_node_ptr_->ActToDropout();
  }

  // Transitions from kDropout state to kPrime and updates current_act_state_
  void DropoutToPrime() final {
    sigmoid_node_ptr_->DropoutToPrime();
  }

  // Transitions from current_act_state_ to state
  bool Transition(ActStates state) final {
    return sigmoid_node_ptr_->Transition(state);
  }

 private:
  SigNodeUPtr<T> sigmoid_node_ptr_{nullptr};

};

// Alias for unique SigCENode pointer
template <class T>
using SigCENodeUPtr = std::unique_ptr<SigCENode<T>>;
 
}  // namespace intellgraph
 
# endif  // INTELLGRAPH_NODE_SIGMOID_CROSS_ENTROPY_H_
 