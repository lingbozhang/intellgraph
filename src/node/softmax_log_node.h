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
#ifndef INTELLGRAPH_NODE_SOFTMAX_LOG_NODE_H_
#define INTELLGRAPH_NODE_SOFTMAX_LOG_NODE_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "node/node_parameter.h"
#include "node/output_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {

template<class T>
class SoftmaxLogNode : public OutputNode<T> {
 public:
  SoftmaxLogNode() noexcept = delete;

  explicit SoftmaxLogNode(REF const NodeParameter& node_param);

  // Move constructor
  SoftmaxLogNode(MOVE SoftmaxLogNode<T>&& rhs) noexcept = default;

  // Move operator
  SoftmaxLogNode& operator=(MOVE SoftmaxLogNode<T>&& rhs) noexcept = default;

  // Copy constructor and operator are explicitly deleted
  SoftmaxLogNode(REF const SoftmaxLogNode<T>& rhs) = delete;
  SoftmaxLogNode& operator=(REF const SoftmaxLogNode<T>& rhs) = delete;

  ~SoftmaxLogNode() noexcept final = default;

  COPY inline std::vector<size_t> get_dims() const final {
    return node_param_.ref_dims();
  }

  REF inline const std::vector<size_t>& ref_dims() const final {
    return node_param_.ref_dims();
  }

  // Accessable operations for the activation matrix
  MUTE inline MatXX<T>* get_activation_ptr() final {
    return &activation_;
  }

  // Accessable operations for the bias vector
  MUTE inline VecX<T>* get_bias_ptr() const final {
    return bias_ptr_.get();
  }
  // Accessable operations for the delta matrix
  MUTE inline MatXX<T>* get_delta_ptr() final {
    return &delta_;
  }
  // Accessable operations for the node parameter
  REF inline const NodeParameter& ref_node_param() const final {
    return node_param_;
  }

  inline void set_activation(COPY T value) final {
    activation_.array() = value;
  }

  void InitializeBias(REF const std::function<T(T)>& functor) final;

  void PrintBias() const final;

  bool CallActFxn() final;

  bool CalcActPrime() final;

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels);

  inline bool ResetActState() final {
    return Transition(kInit);
  }

  void FeedFeature(REF const Eigen::Ref<const MatXX<T>>& feature) final {
    activation_ = feature;
    Transition(kFeed);
  }

  inline void TurnDropoutOn(T dropout_p) final {
    dropout_on_ = true;
    CHECK_GT(dropout_p, 1.0) << "TurnDropoutOn() for SoftmaxLogNode is failed.";
    dropout_p_ = dropout_p;
  }

  inline void TurnDropoutOff() final {
    dropout_on_ = false;
    dropout_p_ = 1.0;
  }

  COPY T CalcLoss(REF const Eigen::Ref<const MatXX<T>>& labels) final;

  bool CalcDelta(REF const Eigen::Ref<const MatXX<T>>& labels) final;
  
 protected:
  // Transitions from kInit to kAct and updates current_act_state_
  void InitToAct() final;

  // Transitions from kAct state to kDropout state and updates current_act_state_
  void ActToDropout() final;

  // Transitions from kDropout state to kPrime and updates current_act_state_
  void DropoutToPrime() final;

  // Transitions from current_act_state_ to state
  bool Transition(ActStates state) final;

 private:
  NodeParameter node_param_{};

  MatXX<T> activation_{};

  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXX<T> delta_{};

  VecXUPtr<T> bias_ptr_{nullptr};

  // Stores current state of activation vector
  ActStates current_act_state_{kInit};

  // A dropout flag
  bool dropout_on_{false};

  T dropout_p_{1.0};

};

// Alias for unique SoftmaxLogNode pointer
template <class T>
using SMLogNodeUPtr = std::unique_ptr<SoftmaxLogNode<T>>;
 
}  // namespace intellgraph
 
# endif  // INTELLGRAPH_NODE_SOFTMAX_LOG_NODE_H_
 