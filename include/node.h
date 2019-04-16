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
#ifndef INTELLGRAPH_NODE_NODE_H_
#define INTELLGRAPH_NODE_NODE_H_
 
#include <functional>
#include <memory>
#include <vector>

#include "glog/logging.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {

// Forward declaration
template <class T>
class ActState;

template <class T>
using ActStateUPtr = std::unique_ptr<ActState<T>>;

template <class T>
class InitialState;

template <class T>
class FeedState;

template <class T>
class ActivateState;

template <class T>
class DropoutState;

template <class T>
class PrimeState;

// In IntellGraph, node is a basic building block that represents a neural
// network layer, all node class in /node directory should implement node class 
// as a base class. In Node classes, in order to save memory, only one 
// matrix (activation matrix) is used to store all weighted_sum, activation, and 
// activation prime results, hence, a state pattern is implemented to control 
// the state transitions of the activation matrix.
template <class T>
class Node {
 public:
  Node() noexcept {}

  explicit Node(REF const NodeParameter& node_param);

  // Move constructor
  Node(MOVE Node<T>&& rhs) = default;

  // Move operator
  Node& operator=(MOVE Node<T>&& rhs) = default;
  
  // Copy constructor and operator are explicitly deleted
  Node(REF const Node<T>& rhs) = delete;
  Node& operator=(REF const Node<T>& rhs) = delete;

  // All interfaces/abstract classes should have virtual destructor in order to 
  // release memory of derived object from interfaces/abstract classes
  virtual ~Node() noexcept = default;

  REF virtual inline const size_t ref_node_id() const {
    return node_param_.ref_id();
  }

  COPY virtual inline std::vector<size_t> get_dims() const {
    return node_param_.ref_dims();
  }

  REF virtual inline const std::vector<size_t>& ref_dims() const {
    return node_param_.ref_dims();
  }

  // Accessable operations for the activation matrix
  MUTE virtual inline MatXX<T>* get_activation_ptr() {
    return &activation_;
  }

  // Accessable operations for the bias vector
  MUTE virtual inline VecX<T>* get_bias_ptr() const {
    return bias_ptr_.get();
  }

  // Accessable operations for the delta matrix
  MUTE virtual inline MatXX<T>* get_delta_ptr() {
    return &delta_;
  }

  // Accessable operations for the node parameter
  REF virtual inline const NodeParameter& ref_node_param() const {
    return node_param_;
  }

  COPY virtual inline const bool ref_dropout_on() const {
    return dropout_on_;
  }

  virtual inline void InitializeBias(REF const std::function<T(T)>& functor) {
    if (functor == nullptr) {
      LOG(WARNING) << "InitializeBias() for Node is failed: " 
                   << "initializes bias with standard normal distribution";
      bias_ptr_->array() = bias_ptr_->array().unaryExpr(std::function<T(T)>( \
          NormalFunctor<T>(0.0, 1.0)));
    } else {
      bias_ptr_->array() = bias_ptr_->array().unaryExpr(functor);
    }
  }

  virtual inline void PrintBias() const {
    std::cout << "Node: " << node_param_.ref_id() << std::endl 
              << "Bias Vector:" << std::endl << bias_ptr_->array() 
              << std::endl;
  }

  virtual inline void TurnDropoutOn(T dropout_p) {
    dropout_on_ = true;
    CHECK_GT(1.0, dropout_p) << "TurnDropoutOn() for Node is failed.";
    dropout_p_ = dropout_p;
  }

  virtual inline void TurnDropoutOff() {
    dropout_on_ = false;
    dropout_p_ = 1.0;
  }

  virtual inline void ToInit() {
    state_ptr_->ToInit(this);
  }

  virtual inline void ToFeed() {
    state_ptr_->ToFeed(this);
  }

  virtual inline void ToAct() {
    state_ptr_->ToAct(this);
  }

  virtual inline void ToDropout() {
    state_ptr_->ToDropout(this);
  }

  virtual inline void ToPrime() {
    state_ptr_->ToPrime(this);
  }

  virtual void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) = 0;

  virtual inline void FeedFeature(REF const Eigen::Ref<const MatXX<T>>& feature) {
    activation_ = feature;
    state_ptr_->ToFeed(this);
  }

 protected:
  friend class ActState<T>;
  friend class InitialState<T>;
  friend class FeedState<T>;
  friend class ActivateState<T>;
  friend class DropoutState<T>;
  friend class PrimeState<T>;

  inline void ChangeState(MOVE ActStateUPtr<T> state_ptr) {
    state_ptr_ = std::move(state_ptr);
  }

  virtual void Activate() = 0;

  virtual inline void Dropout() {
    activation_.array().unaryExpr(std::function<T(T)>( \
        BernoulliFunctor<T>(dropout_p_)));
  }

  virtual void Prime() = 0;

  COPY virtual std::string get_node_state() {
    return state_ptr_->get_state_name();
  }

 private:
  NodeParameter node_param_{};

  MatXX<T> activation_{};

  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXX<T> delta_{};

  VecXUPtr<T> bias_ptr_{nullptr};

  // A dropout flag
  bool dropout_on_{false};

  T dropout_p_{1.0};

  ActStateUPtr<T> state_ptr_;

};
// Alias for unique node pointer
template <class T>
using NodeUPtr = std::unique_ptr<Node<T>>;

// State Pattern for the Node class. Note ToInit() method does not change the
// activation matrix, since we will have multiple initialization implementations
template <class T>
class ActState {
 public:
  ActState() {}

  virtual ~ActState() = default;

  virtual std::string get_state_name() {return "";}

  virtual void ToInit(MUTE Node<T>* node) {
    LOG(WARNING) << "Node: " << node->ref_node_id()
                 << ", ToInit() is not defined in current state: "
                 << node->get_node_state();
  }

  virtual void ToFeed(MUTE Node<T>* node) {
    LOG(WARNING) << "Node: " << node->ref_node_id()
                 << ", ToFeed() is not defined in current state: "
                 << node->get_node_state();
  }

  virtual void ToAct(MUTE Node<T>* node) {
    LOG(WARNING) << "Node: " << node->ref_node_id()
                 << ", ToAct() is not defined in current state: "
                 << node->get_node_state();
  }

  virtual void ToPrime(MUTE Node<T>* node) {
    LOG(WARNING) << "Node: " << node->ref_node_id()
                 << ", ToPrime() that is not defined in current state: "
                 << node->get_node_state();
  }

  virtual void ToDropout(MUTE Node<T>* node) {
    LOG(WARNING) << "Node: " << node->ref_node_id()
                 << ", ToDropout() is not defined in current state: "
                 << node->get_node_state();
  }

 protected:
  void ChangeState(MUTE Node<T>* node, MOVE ActStateUPtr<T> state);

};

template <class T>
class InitialState : implements ActState<T> {
 public:
  InitialState() {};

  static ActStateUPtr<T> get_instance() {
    return std::make_unique<InitialState<T>>();
  }

  ~InitialState() {};

  COPY inline std::string get_state_name() {
    return "InitialState";
  }

  void ToInit(MUTE Node<T>* node) final;

  void ToFeed(MUTE Node<T>* node) final;

  void ToAct(MUTE Node<T>* node) final;

};

template <class T>
class FeedState : implements ActState<T> {
 public:
  FeedState() {};

  static ActStateUPtr<T> get_instance() {
    return std::make_unique<FeedState<T>>();
  }

  ~FeedState() {};

  COPY inline std::string get_state_name() {
    return "FeedState";
  }

  void ToInit(MUTE Node<T>* node) final;

  void ToAct(MUTE Node<T>* node) final;

  void ToPrime(MUTE Node<T>* node) final;

  void ToFeed(MUTE Node<T>* node) final;

};

template <class T>
class ActivateState : implements ActState<T> {
 public:
  ActivateState() {};

  static ActStateUPtr<T> get_instance() {
    return std::make_unique<ActivateState<T>>();
  }

  ~ActivateState() {};

  COPY inline std::string get_state_name() {
    return "ActivateState";
  }

  void ToInit(MUTE Node<T>* node) final;

  void ToPrime(MUTE Node<T>* node) final;

  void ToDropout(MUTE Node<T>* node) final;

};

template <class T>
class DropoutState : implements ActState<T> {
 public:
  DropoutState() {};

  static ActStateUPtr<T> get_instance() {
    return std::make_unique<DropoutState<T>>();
  }

  ~DropoutState() {};

  COPY inline std::string get_state_name() {
    return "DropoutState";
  }

  void ToPrime(MUTE Node<T>* node) final;

};

template <class T>
class PrimeState : implements ActState<T> {
 public:
  PrimeState() {};

  static ActStateUPtr<T> get_instance() {
    return std::make_unique<PrimeState<T>>();
  }

  ~PrimeState() {};

  COPY inline std::string get_state_name() {
    return "PrimeState";
  }

  void ToInit(MUTE Node<T>* node) final;

};

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_NODE_H_







  