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
#ifndef INTELLGRAPH_NODE_SIGMOID_L2_NODE_H_
#define INTELLGRAPH_NODE_SIGMOID_L2_NODE_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "node/node_parameter.h"
#include "node/output_node.h"
#include "node/sigmoid_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// SigL2Node uses decorator pattern (see https://dzone.com/articles/is-inheritance-dead)
// SigL2Node improves performance of GetLoss, CalcDelta, GetLoss, and CalcDelta 
// with Eigen library. In SigL2Node, sigmoid function is used as a activation 
// function and squared Euclidean norm is used as a loss function.
template<class T>
class SigL2Node : public OutputNode<T> {
 public:
  SigL2Node() noexcept = delete;

  explicit SigL2Node(REF const NodeParameter& node_param) {
    node_ptr_ = std::make_unique<SigmoidNode<T>>(node_param);
  }

  // Move constructor
  SigL2Node(MOVE SigL2Node<T>&& rhs) = default;

  // Move operator
  SigL2Node& operator=(MOVE SigL2Node<T>&& rhs) = default;

  // Copy constructor and operator are explicitly deleted
  SigL2Node(REF const SigL2Node<T>& rhs) = delete;
  SigL2Node& operator=(REF const SigL2Node<T>& rhs) = delete;

  ~SigL2Node() noexcept final = default;

  COPY T CalcLoss(REF const Eigen::Ref<const MatXX<T>>& labels) final;

  bool CalcDelta(REF const Eigen::Ref<const MatXX<T>>& labels) final;

  REF inline const size_t ref_node_id() const final {
    return node_ptr_->ref_node_id();
  }

  COPY inline std::vector<size_t> get_dims() const final {
    return node_ptr_->get_dims();
  }

  REF inline const std::vector<size_t>& ref_dims() const final {
    return node_ptr_->ref_dims();
  }

  // Accessable operations for the activation matrix
  MUTE inline MatXX<T>* get_activation_ptr() final {
    return node_ptr_->get_activation_ptr();
  }

  // Accessable operations for the bias vector
  MUTE inline VecX<T>* get_bias_ptr() const final {
    return node_ptr_->get_bias_ptr();
  }

  // Accessable operations for the delta matrix
  MUTE inline MatXX<T>* get_delta_ptr() final {
    return node_ptr_->get_delta_ptr();
  }

  // Accessable operations for the node parameter
  REF inline const NodeParameter& ref_node_param() const final {
    return node_ptr_->ref_node_param();
  }

  COPY inline const bool ref_dropout_on() const final {
    return node_ptr_->ref_dropout_on();
  }

  COPY std::string get_node_state() final {
    return node_ptr_->get_node_state();
  }

  REF inline const T ref_dropout_p() const final {
    return node_ptr_->ref_dropout_p();
  }
  inline void InitializeBias(REF const std::function<T(T)>& functor) final {
    node_ptr_->InitializeBias(functor);
  }

  inline void PrintBias() const final {
    node_ptr_->PrintBias();
  }

  inline void TurnDropoutOn(T dropout_p) final {
    node_ptr_->TurnDropoutOn(dropout_p);
  }

  inline void TurnDropoutOff() final {
    node_ptr_->TurnDropoutOff();
  }

  inline void ToInit() final {
    node_ptr_->ToInit();
  }

  inline void ToFeed() override {
    node_ptr_->ToFeed();
  }

  inline void ToAct() final {
    node_ptr_->ToAct();
  }

  inline void ToDropout() final {
    node_ptr_->ToDropout();
  }

  inline void ToPrime() final {
    node_ptr_->ToPrime();
  }

  virtual void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) final {
    node_ptr_->Evaluate(labels);
  }

  inline void FeedFeature(REF const Eigen::Ref<const MatXX<T>>& feature) final {
    node_ptr_->FeedFeature(feature);
    node_ptr_->ToFeed();
  }

 protected:
  void Activate() final {
    node_ptr_->Activate();
  }

  void Prime() final {
    node_ptr_->Prime();
  }

 private:
  SigNodeUPtr<T> node_ptr_{nullptr};

};

// Alias for unique SigL2Node pointer
template <class T>
using SigL2NodeUPtr = std::unique_ptr<SigL2Node<T>>;
 
}  // namespace intellgraph
 
# endif  // INTELLGRAPH_NODE_SIGMOID_L2_NODE_H_
 