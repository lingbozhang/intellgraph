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
#include "node/act_loss_node.h"

namespace intellgraph {

template <class T>
ActLossNode<T>::ActLossNode(
    const NodeParameter& node_param,
    const std::function<T(T)>& act_function_ptr,
    const std::function<T(T)>& act_prime_ptr,
    const std::function<T(const MatXX<T>&, const MatXX<T>&)>& loss_function_ptr,
    const std::function<void(const MatXX<T>&, const MatXX<T>&, MatXX<T>&)>& \
        loss_prime_ptr)
    : act_function_ptr_(act_function_ptr), act_prime_ptr_(act_prime_ptr), \
      loss_function_ptr_(loss_function_ptr), loss_prime_ptr_(loss_prime_ptr) {
  node_param_.Clone(node_param);

  size_t row = node_param.get_k_dims()[0];
  size_t col = node_param.get_k_dims()[1];

  activation_ptr_ = std::make_unique<MatXX<T>>(row, col);
  delta_ptr_ = std::make_unique<MatXX<T>>(row, col);
  bias_ptr_ = std::make_unique<MatXX<T>>(row, col);

  activation_ptr_->array() = 0.0;
  delta_ptr_->array() = 0.0;
  bias_ptr_->array() = 0.0;
  
  Transition(kInit);
}

template <class T>
void ActLossNode<T>::PrintAct() const {
  std::cout << "ActLossNode " << node_param_.get_k_id() << " Activation Vector:" 
            << std::endl << activation_ptr_->array() << std::endl;
}

template <class T>
void ActLossNode<T>::PrintDelta() const {
  std::cout << "ActLossNode " << node_param_.get_k_id() << " Delta Vector:" 
            << std::endl << delta_ptr_->array() << std::endl;
}

template <class T>
void ActLossNode<T>::PrintBias() const {
  std::cout << "ActLossNode " << node_param_.get_k_id() << " Bias Vector:" 
            << std::endl << bias_ptr_->array() << std::endl;
}

template <class T>
void ActLossNode<T>::ApplyUnaryFunctor_k(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    std::cout << "WARNING: functor passed to ApplyUnaryFunctor() is not defined." 
              << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array(). \
                               unaryExpr(functor);
    Transition(kInit);
  }
}

template <class T>
T ActLossNode<T>::CalcLoss_k(const MatXX<T>& data_result) {
  T loss = 0;
  if (!Transition(kAct)) {
    std::cout << "ERROR: CalcLoss() for ActLossNode fails. " 
              << "Transition to kAct fails" << std::endl;
    exit(1);
  }
  if (loss_function_ptr_ == nullptr) {
    std::cout << "WARNING: loss function is not defined." << std::endl;
  } else {
    loss = loss_function_ptr_(*activation_ptr_, data_result);
  }
  return loss;
}

template <class T>
void ActLossNode<T>::CalcDelta_k(const MatXX<T>& data_result) {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CalcDelta() for ActLossNode fails. " 
              << "Transition to kAct fails" << std::endl;
    exit(1);
  }
  if (loss_prime_ptr_ == nullptr) {
    std::cout << "WARNING: loss prime function is not defined." << std::endl;
  } else {
    loss_prime_ptr_(*activation_ptr_, data_result, *delta_ptr_);
  }
  // Note CalcActPrime overwrites data in activation_ptr in-place
  if (!Transition(kPrime)) {
    std::cout << "ERROR: CalcDelta() for ActLossNode fails. " 
              << "Transition to kPrime fails" << std::endl;
    exit(1);
  }
  delta_ptr_->array() *= activation_ptr_->array();
}

// Transitions from kInit state to kAct state. 
template <class T>
void ActLossNode<T>::InitToAct() {
  if (act_function_ptr_ == nullptr) {
    std::cout << "WARNING: activation function is not defined." << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array(). \
                               unaryExpr(act_function_ptr_);
  }
  current_act_state_ = kAct;
}

template <class T>
void ActLossNode<T>::ActToPrime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  if (act_prime_ptr_ == nullptr) {
    std::cout << "WARNING: activation prime function is not defined."
              << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array(). \
                               unaryExpr(act_prime_ptr_);
  }
  current_act_state_ = kPrime;
}

template <class T>
bool ActLossNode<T>::Transition(ActStates state) {
  if (state == kInit) {
    current_act_state_ = kInit;
    return true;
  }
  if (current_act_state_ > state) {
    std::cout << "ERROR: Transition() for ActivationNode fails" << std::endl;
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
        std::cout << "ERROR: Transition() for ActivationNode fails to handle"
                  << "input state" << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <class T>
void ActLossNode<T>::CallActFxn() {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CallActFxn() for ActLossNode fails" << std::endl;
    exit(1);
  }
}

template <class T>
void ActLossNode<T>::CalcActPrime() {
  if (!Transition(kPrime)) {
    std::cout << "ERROR: CalcActPrime() for ActLossNode fails" << std::endl;
    exit(1);
  }
}

// Instantiate class, otherwise compilation will fail
template class ActLossNode<float>;
template class ActLossNode<double>;

}  // namespace intellgraph