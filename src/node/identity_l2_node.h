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
#ifndef INTELLGRAPH_NODE_IDENTITY_L2_NODE_H_
#define INTELLGRAPH_NODE_IDENTITY_L2_NODE_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "node/node_parameter.h"
#include "node/output_node.h"
#include "node/identity_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

template<class T>
class IDL2Node : public OutputNode<T> {
 public:
  IDL2Node() noexcept = delete;

  explicit IDL2Node(REF const NodeParameter& node_param) {
    identity_node_ptr_ = std::make_unique<IdentityNode<T>>(node_param);
  }

  // Move constructor
  IDL2Node(MOVE IDL2Node<T>&& rhs) noexcept = default;

  // Move operator
  IDL2Node& operator=(MOVE IDL2Node<T>&& rhs) noexcept = default;

  // Copy constructor and operator are explicitly deleted
  IDL2Node(REF const IDL2Node<T>& rhs) = delete;
  IDL2Node& operator=(REF const IDL2Node<T>& rhs) = delete;

  ~IDL2Node() noexcept final = default;

  COPY inline std::vector<size_t> get_dims() const final {
    return identity_node_ptr_->get_dims();
  }

  REF inline const std::vector<size_t>& ref_dims() const final {
    return identity_node_ptr_->ref_dims();
  }

  // Accessable operations for the activation matrix
  MUTE inline MatXX<T>* get_activation_ptr() final {
    return identity_node_ptr_->get_activation_ptr();
  }

  // Accessable operations for the bias vector
  MUTE inline VecX<T>* get_bias_ptr() const final {
    return identity_node_ptr_->get_bias_ptr();
  }
  // Accessable operations for the delta matrix
  MUTE inline MatXX<T>* get_delta_ptr() final {
    return identity_node_ptr_->get_delta_ptr();
  }
  // Accessable operations for the node parameter
  REF inline const NodeParameter& ref_node_param() const final {
    return identity_node_ptr_->ref_node_param();
  }

  inline void set_activation(COPY T value) final {
    identity_node_ptr_->set_activation(value);
  }

  void InitializeBias(REF const std::function<T(T)>& functor) final {
    identity_node_ptr_->InitializeBias(functor);
  }

  void PrintBias() const final {
    identity_node_ptr_->PrintBias();
  }

  bool CallActFxn() final {
    return identity_node_ptr_->CallActFxn();
  }

  bool CalcActPrime() final {
    return identity_node_ptr_->CalcActPrime();
  }

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) {
    identity_node_ptr_->Evaluate(labels);
  }

  inline bool ResetActState() final {
    return identity_node_ptr_->ResetActState();
  }

  void FeedFeature(REF const Eigen::Ref<const MatXX<T>>& feature) final {
    identity_node_ptr_->FeedFeature(feature);
  }

  COPY T CalcLoss(REF const Eigen::Ref<const MatXX<T>>& labels) final;

  bool CalcDelta(REF const Eigen::Ref<const MatXX<T>>& labels) final;
  
 protected:
  // Transitions from kAct state to kPrime state and updates current_act_state_
  void ActToPrime() final {
    identity_node_ptr_->ActToPrime();
  }

  // Transitions from kInit state to kAct state and updates current_act_state_
  void InitToAct() final {
    identity_node_ptr_->InitToAct();
  }

  // Transitions from current_act_state_ to state
  bool Transition(ActStates state) final {
    return identity_node_ptr_->Transition(state);
  }

 private:
  IDNodeUPtr<T> identity_node_ptr_{nullptr};

};

// Alias for unique IDL2Node pointer
template <class T>
using IDL2NodeUPtr = std::unique_ptr<IDL2Node<T>>;
 
}  // namespace intellgraph
 
# endif  // INTELLGRAPH_NODE_IDENTITY_L2_NODE_H_
 