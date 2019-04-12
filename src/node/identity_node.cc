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
#include "node/identity_node.h"

namespace intellgraph {

template <class T>
IdentityNode<T>::IdentityNode(const NodeParameter& node_param) {
  node_param_.Clone(node_param);

  size_t row = node_param.ref_dims()[0];
  size_t col = node_param.ref_dims()[1];

  bias_ptr_ = std::make_unique<VecX<T>>(row);

  bias_ptr_->array() = 0.0;

  current_act_state_ = kInit;
}

template <class T>
void IdentityNode<T>::PrintBias() const {
  std::cout << "Node: " << node_param_.ref_id() << std::endl 
            << "Bias Vector:" << std::endl << bias_ptr_->array() 
            << std::endl;
}

template <class T>
void IdentityNode<T>::InitializeBias(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    LOG(WARNING) << "InitializeBias() for IdentityNode is failed: " 
                 << "initializes bias with standard normal distribution";
    bias_ptr_->array() = bias_ptr_->array().unaryExpr(std::function<T(T)>( \
        NormalFunctor<T>(0.0, 1.0)));
  } else {
    bias_ptr_->array() = bias_ptr_->array().unaryExpr(functor);
  }
  Transition(kInit);
}

// Transitions from kInit state to kAct state. In order to avoid overflow of 
// exp() function, Identity function is calculated based on the sign of 
// activation vector entry, as shown in the implementation below.
template <class T>
void IdentityNode<T>::InitToAct() {
  // Identity activation function:
  current_act_state_ = kAct;  
}

template <class T>
void IdentityNode<T>::ActToPrime() {
  // Derivative equation:
  activation_.array() = 1.0;
  current_act_state_ = kPrime;
}

template <class T>
bool IdentityNode<T>::Transition(ActStates state) {
  if (state == kFeed) {
    current_act_state_ = state;
    return true;
  }

  // Nothing happens if current node is an input node.
  // Note, an internal node permanently changes to an input node
  // when Transition(kFeed) is called
  if (current_act_state_ == kFeed) {
    return true;
  }

  if (state == kInit) {
    current_act_state_ = state;
    return true;
  }

  if (current_act_state_ > state) {
    LOG(ERROR) << "Transition() for IdentityNode is failed: "
               << "current state: " << current_act_state_
               << ", transition state: " << state;
    return false;
  }

  while (current_act_state_ < state) {
    switch (current_act_state_) {
      case kInit: {
        InitToAct();
        break;
      }
      case kAct: {
        ActToPrime();
        break;
      }
      default: {
        LOG(ERROR) << "Transition() for IdentityNode is failed";
        return false;
      }
    }
  }
  return true;
}

template <class T>
bool IdentityNode<T>::CallActFxn() {
  if (!Transition(kAct)) {
    LOG(ERROR) << "CallActFxn() for IdentityNode is failed";
    return false;
  }
  return true;
}

template <class T>
bool IdentityNode<T>::CalcActPrime() {
  if (!Transition(kPrime)) {
    LOG(ERROR) << "CalcActPrime() for IdentityNode is failed";
    return false;
  }
  return true;
}

template <class T>
void IdentityNode<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& labels) {
  if (!Transition(kAct)) {
    LOG(ERROR) << "Evaluate() for IdentityNode is failed.";
    exit(1);
  }

  CHECK_EQ(activation_.cols(), labels.cols())
      << "CalcLoss() for IdentityNode is failed: "
      << "activation and data matrix dimensions are not equal!";

  T loss = (get_activation_ptr()->array() - labels.array()). \
            matrix().norm();
  T avg_norm = loss / labels.cols();
  std::cout << "Average l2 norm: " << avg_norm << std::endl;
}

// Instantiate class, otherwise compilation will fail
template class IdentityNode<float>;
template class IdentityNode<double>;

}  // namespace intellgraph









