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
#include "node/sigmoid_node.h"

namespace intellgraph {

template <class T>
SigmoidNode<T>::SigmoidNode(const NodeParameter<T>& node_param) {
    node_param_.Clone(node_param);
    size_t row = node_param.ref_dims()[0];
    size_t col = node_param.ref_dims()[1];
    activation_ptr_ = std::make_unique<MatXX<T>>(row, col);
    delta_ptr_ = std::make_unique<MatXX<T>>(row, col);
    bias_ptr_ = std::make_unique<MatXX<T>>(row, col);

    activation_ptr_->array() = 0.0;
    delta_ptr_->array() = 0.0;
    bias_ptr_->array() = 0.0;

    current_act_state_ = kInit;
}

template <class T>
void SigmoidNode<T>::PrintAct() const {
  std::cout << "Node: " << node_param_.ref_id() << std::endl
            << " Activation Vector:" << std::endl << activation_ptr_->array() 
            << std::endl;
}

template <class T>
void SigmoidNode<T>::PrintDelta() const {
  std::cout << "Node: " << node_param_.ref_id() << std::endl
            << " Delta Vector:" << std::endl << delta_ptr_->array() 
            << std::endl;
}

template <class T>
void SigmoidNode<T>::PrintBias() const {
  std::cout << "Node: " << node_param_.ref_id() << std::endl 
            << " Bias Vector:" << std::endl << bias_ptr_->array() 
            << std::endl;
}

template <class T>
void SigmoidNode<T>::InitializeAct(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    LOG(WARNING) << "InitializeAct() for SigmoidNode is failed: "
                 << "Initializes activation with standard normal distribution";
    activation_ptr_->array() = activation_ptr_->array().unaryExpr( \
        std::function<T(T)>(NormalFunctor<T>(0.0, 1.0)));
  } else {
    activation_ptr_->array() = activation_ptr_->array().unaryExpr(functor);
  }
  Transition(kInit);
}

template <class T>
void SigmoidNode<T>::InitializeBias(const std::function<T(T)>& functor) {
  VecX<T> vec(bias_ptr_->array().rows());
  if (functor == nullptr) {
    LOG(WARNING) << "InitializeBias() for SigmoidNode is failed:" 
                 << "Initializes bias with standard normal distribution";
    vec.array() = vec.array().unaryExpr(std::function<T(T)>( \
        NormalFunctor<T>(0.0, 1.0)));
  } else {
    vec.array() = vec.array().unaryExpr(functor);
  }
  bias_ptr_->matrix().colwise() = vec;
  Transition(kInit);
}

// Transitions from kInit state to kAct state. In order to avoid overflow of 
// exp() function, sigmoid function is calculated based on the sign of 
// activation vector entry, as shown in the implementation below.
template <class T>
void SigmoidNode<T>::InitToAct() {
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
void SigmoidNode<T>::ActToPrime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  activation_ptr_->array() *= (1.0 - activation_ptr_->array());
  current_act_state_ = kPrime;
}

template <class T>
bool SigmoidNode<T>::Transition(ActStates state) {
  if (state == kInit) {
    current_act_state_ = kInit;
    return true;
  }
  if (current_act_state_ > state) {
    LOG(ERROR) << "Transition() for SigmoidNode is failed";
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
        LOG(ERROR) << "Transition() for SigmoidNode is failed";
        return false;
      }
    }
  }
  return true;
}

template <class T>
bool SigmoidNode<T>::CallActFxn() {
  if (!Transition(kAct)) {
    LOG(ERROR) << "CallActFxn() for SigmoidNode is failed";
    return false;
  }
  return true;
}

template <class T>
bool SigmoidNode<T>::CalcActPrime() {
  if (!Transition(kPrime)) {
    LOG(ERROR) << "CalcActPrime() for SigmoidNode is failed";
    return false;
  }
  return true;
}

// Instantiate class, otherwise compilation will fail
template class SigmoidNode<float>;
template class SigmoidNode<double>;

}  // namespace intellgraph









