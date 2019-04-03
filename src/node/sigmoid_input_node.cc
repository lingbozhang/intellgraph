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
#include "node/sigmoid_input_node.h"

namespace intellgraph {

template <class T>
SigInputNode<T>::SigInputNode(const NodeParameter<T>& node_param) {
    node_param_.Clone(node_param);
    size_t row = node_param.get_k_dims()[0];
    size_t col = node_param.get_k_dims()[1];
    activation_ptr_ = std::make_unique<MatXX<T>>(row, col);
    delta_ptr_ = std::make_unique<MatXX<T>>(row, col);
    bias_ptr_ = std::make_unique<MatXX<T>>(row, col);

    activation_ptr_->array() = 0.0;
    delta_ptr_->array() = 0.0;
    bias_ptr_->array() = 0.0;

    current_act_state_ = kInit;
}

template <class T>
void SigInputNode<T>::PrintAct() const {
  std::cout << "SigInputNode: " << node_param_.get_k_id() << " Activation Vector:"
            << std::endl << activation_ptr_->array() << std::endl;
}

template <class T>
void SigInputNode<T>::PrintDelta() const {
  std::cout << "SigInputNode: " << node_param_.get_k_id() << " Delta Vector:"
            << std::endl << delta_ptr_->array() << std::endl;
}

template <class T>
void SigInputNode<T>::PrintBias() const {
  std::cout << "SigInputNode: " << node_param_.get_k_id() << " Bias Vector:"
            << std::endl << bias_ptr_->array() << std::endl;
}

template <class T>
void SigInputNode<T>::ApplyUnaryFunctor_k(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    std::cout << "WARNING: functor passed to ApplyUnaryFunctor() is not defined." 
              << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array().unaryExpr(functor);
    Transition(kInit);
  }
}

// Transitions from kInit state to kAct state. In order to avoid overflow of 
// exp() function, sigmoid function is calculated based on the sign of 
// activation vector entry, as shown in the implementation below.
template <class T>
void SigInputNode<T>::InitToAct() {
  // Sigmoid activation function:
  // f(z)=1.0/(1.0+exp(-z))
  for (size_t i = 0; i < activation_ptr_->array().rows(); ++i) {
    for (size_t j = 0; j < activation_ptr_->array().cols(); ++j) {
      T element_value = activation_ptr_->array()(i, j);
      if (element_value >= 0.0) {
        activation_ptr_->array()(i, j) = 1.0 / (1.0 + std::exp(-element_value));
      } else {
        activation_ptr_->array()(i, j) = std::exp(element_value) / \
                                         (1.0 + std::exp(element_value));
      }
    }
  }
  current_act_state_ = kAct;  
}

template <class T>
void SigInputNode<T>::ActToPrime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  activation_ptr_->array() *= (1.0 - activation_ptr_->array());
  current_act_state_ = kPrime;
}

template <class T>
bool SigInputNode<T>::Transition(ActStates state) {
  if (state == kInit) {
    current_act_state_ = kInit;
    return true;
  }
  if (current_act_state_ > state) {
    std::cout << "ERROR: Transition() for SigInputNode fails" << std::endl;
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
        std::cout << "ERROR: Transition() for SigInputNode fails to handle"
                  << "input state" << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <class T>
void SigInputNode<T>::CallActFxn() {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CallActFxn() for SigInputNode fails" << std::endl;
    exit(1);
  }
}

template <class T>
void SigInputNode<T>::CalcActPrime() {
  if (!Transition(kPrime)) {
    std::cout << "ERROR: CalcActPrime() for SigInputNode fails" << std::endl;
    exit(1);
  }
}

// Instantiate class, otherwise compilation will fail
template class SigInputNode<float>;
template class SigInputNode<double>;

}  // namespace intellgraph








