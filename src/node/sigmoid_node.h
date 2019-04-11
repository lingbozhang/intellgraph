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

#include "glog/logging.h"
#include "node/internal_node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {
 
// Forward declaration
template <class T>
class SigL2Node;

// SigmoidNode improves performance of CallActFxn and CalcActPrime with Eigen 
// library and has better performance than ActivationNode. 
template <class T>
class SigmoidNode : implements IntNode<T> {
 public:
  friend class SigL2Node<T>;

  SigmoidNode() noexcept = default;

  explicit SigmoidNode(REF const NodeParameter& node_param);

  // Move constructor
  SigmoidNode(MOVE SigmoidNode<T>&& rhs) noexcept = default;

  // Move operator
  SigmoidNode& operator=(MOVE SigmoidNode<T>&& rhs) noexcept = default;
  
  // Copy constructor and operator are explicitly deleted
  SigmoidNode(REF const SigmoidNode<T>& rhs) = delete;
  SigmoidNode& operator=(REF const SigmoidNode<T>& rhs) = delete;

  virtual ~SigmoidNode() noexcept = default;

  COPY inline std::vector<size_t> get_dims() const final {
    return node_param_.ref_dims();
  }

  REF inline const std::vector<size_t>& ref_dims() const final {
    return node_param_.ref_dims();
  }

  // Accessable operations for the activation matrix
  MUTE inline MatXX<T>* get_activation_ptr() const final {
    return activation_ptr_.get();
  }

  // Accessable operations for the bias vector
  MUTE inline VecX<T>* get_bias_ptr() const final {
    return bias_ptr_.get();
  }
  // Accessable operations for the delta matrix
  MUTE inline MatXX<T>* get_delta_ptr() const final {
    return delta_ptr_.get();
  }
  // Accessable operations for the node parameter
  REF inline const NodeParameter& ref_node_param() const final {
    return node_param_;
  }

  inline void set_activation(COPY T value) final {
    activation_ptr_->array() = value;
    Transition(kInit);
  }
  
  inline void move_activation_ptr(MOVE MatXXUPtr<T> activation_ptr) final {
    CHECK_EQ(activation_ptr_->size(), activation_ptr->size()) 
        << "move_activation_ptr() for SigmoidNode is failed: ";
    activation_ptr_ = std::move(activation_ptr);
    Transition(kInit);
  }

  // Passes a functor and applies it on the activation matrix
  void InitializeAct(REF const std::function<T(T)>& functor) final;

  inline void move_bias_ptr(MOVE VecXUPtr<T> bias_ptr) final {
    CHECK_EQ(bias_ptr_->size(), bias_ptr->size())
        << "move_bias_ptr() for SigmoidNode is failed";
    bias_ptr_ = std::move(bias_ptr);
  }

  void InitializeBias(REF const std::function<T(T)>& functor) final;

  inline void move_delta_ptr(MOVE MatXXUPtr<T> delta_ptr) final {
    CHECK_EQ(delta_ptr_->size(), delta_ptr->size())
        << "move_delta_ptr() for SigmoidNode is failed";
    delta_ptr_ = std::move(delta_ptr);
  }

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;

  bool CallActFxn() final;

  bool CalcActPrime() final;

  void Evaluate(REF const MatXX<T>* labels_ptr) final;

  bool ResetActState() final {
    return Transition(kInit);
  }

  void FeedFeature(REF const MatXXSPtr<T>& feature_ptr) final {
    activation_ptr_ = feature_ptr;
    Transition(kFeed);
  }

 protected:
  // Transitions from kAct state to kPrime state and updates current_act_state_
  void ActToPrime() final;

  // Transitions from kInit state to kAct state and updates current_act_state_
  void InitToAct() final;

  // Transitions from current_act_state_ to state
  bool Transition(ActStates state) final;

 private:
  NodeParameter node_param_{};

  MatXXSPtr<T> activation_ptr_{nullptr};

  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXUPtr<T> delta_ptr_{nullptr};

  VecXUPtr<T> bias_ptr_{nullptr};

  // Stores current state of activation vector
  ActStates current_act_state_{kInit};

};

// Alias for unqiue SigmoidNode pointer
template <class T>
using SigNodeUPtr = std::unique_ptr<SigmoidNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_LAYER_SIGMOID_NODE_H_







