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
#include "node/sigmoid_l2_node.h"

namespace intellgraph {

template <class T>
SigL2Node<T>::SigL2Node(const NodeParameter& node_param)
      : node_param_(node_param) {
    activation_ptr_ = std::make_shared<MatXX<T>>(node_param.dims[0], 1);
    delta_ptr_ = std::make_shared<MatXX<T>>(node_param.dims[0], 1);
    bias_ptr_ = std::make_shared<MatXX<T>>(node_param.dims[0], 1);

    activation_ptr_->array() = 0.0;
    delta_ptr_->array() = 0.0;
    bias_ptr_->array() = 0.0;

    Transition(kInit);
}

template <class T>
void SigL2Node<T>::PrintAct() const {
  std::cout << "SigL2Node " << node_param_.id << " Activation Vector:" 
            << std::endl << activation_ptr_->array() << std::endl;
}

template <class T>
void SigL2Node<T>::PrintDelta() const {
  std::cout << "SigL2Node " << node_param_.id << " Delta Vector:" 
            << std::endl << delta_ptr_->array() << std::endl;
}

template <class T>
void SigL2Node<T>::PrintBias() const {
  std::cout << "SigL2Node " << node_param_.id << " Bias Vector:" 
            << std::endl << bias_ptr_->array() << std::endl;
}

template <class T>
void SigL2Node<T>::ApplyUnaryFunctor(std::function<T(T)> functor) {
  if (functor == nullptr) {
    std::cout << "WARNING: functor passed to ApplyUnaryFunctor() is not defined." 
              << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array().unaryExpr(functor);
    Transition(kInit);
  }
}

template <class T>
T SigL2Node<T>::CalcLoss(MatXXSPtr<T>& data_result) {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CalcDelta() for SigL2Node fails. " << std::endl;
    exit(1);
  }
  T loss = (activation_ptr_->array() - data_result->array()). \
            matrix().squaredNorm();
  return loss;
}

template <class T>
void SigL2Node<T>::CalcDelta(MatXXSPtr<T>& data_result) {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CalcDelta() for SigL2Node fails. " 
              << "Transition to kAct fails" << std::endl;
    exit(1);
  }
  delta_ptr_->array() = 2.0 * (activation_ptr_->array() - data_result->array());
  // Note CalcActPrime overwrites data in activation_ptr_ in-place
  if (!Transition(kPrime)) {
    std::cout << "ERROR: CalcDelta() for SigL2Node fails. "
              << "Transition to kPrime fails" << std::endl;
    exit(1);
  } 
  delta_ptr_->array() *= activation_ptr_->array();
}

// Transitions from kInit state to kAct state. In order to avoid overflow of 
// exp() function, sigmoid function is calculated based on the sign of 
// activation vector entry, as shown in the implementation below.
template <class T>
void SigL2Node<T>::InitToAct() {
  // Sigmoid activation function:
  // f(z)=1.0/(1.0+exp(-z))
  for (size_t i = 0; i < activation_ptr_->array().rows(); ++i) {
    T element_value = activation_ptr_->array()(i);
    if (element_value >= 0.0) {
      activation_ptr_->array()(i) = 1.0 / (1.0 + std::exp(-element_value));
    } else {
      activation_ptr_->array()(i) = std::exp(element_value) / \
                                    (1.0 + std::exp(element_value));
    }
  }
  current_act_state_ = kAct;  
}

template <class T>
void SigL2Node<T>::ActToPrime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  activation_ptr_->array() *= (1.0 - activation_ptr_->array());
  current_act_state_ = kPrime;
}

template <class T>
bool SigL2Node<T>::Transition(ActStates state) {
  if (state == kInit) {
    current_act_state_ = kInit;
    return true;
  }
  if (current_act_state_ > state) {
    std::cout << "ERROR: Transition() for SigL2Node fails" << std::endl;
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
        std::cout << "ERROR: Transition() for SigL2Node fails to handle"
                  << "input state" << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <class T>
void SigL2Node<T>::CallActFxn() {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CallActFxn() for SigL2Node fails" << std::endl;
    exit(1);
  }
}

template <class T>
void SigL2Node<T>::CalcActPrime() {
  if (!Transition(kPrime)) {
    std::cout << "ERROR: CalcActPrime() for SigL2Node fails" << std::endl;
    exit(1);
  }
}

// Instantiate class, otherwise compilation will fail
template class SigL2Node<float>;
template class SigL2Node<double>;

}  // namespace intellgraph