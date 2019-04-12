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
SigmoidNode<T>::SigmoidNode(const NodeParameter& node_param) {
  node_param_.Clone(node_param);

  size_t row = node_param.ref_dims()[0];
  size_t col = node_param.ref_dims()[1];

  bias_ptr_ = std::make_unique<VecX<T>>(row);

  bias_ptr_->array() = 0.0;

  current_act_state_ = kInit;
}

template <class T>
void SigmoidNode<T>::PrintBias() const {
  std::cout << "Node: " << node_param_.ref_id() << std::endl 
            << "Bias Vector:" << std::endl << bias_ptr_->array() 
            << std::endl;
}

template <class T>
void SigmoidNode<T>::InitializeBias(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    LOG(WARNING) << "InitializeBias() for SigmoidNode is failed: " 
                 << "initializes bias with standard normal distribution";
    bias_ptr_->array() = bias_ptr_->array().unaryExpr(std::function<T(T)>( \
        NormalFunctor<T>(0.0, 1.0)));
  } else {
    bias_ptr_->array() = bias_ptr_->array().unaryExpr(functor);
  }
  Transition(kInit);
}

// Transitions from kInit state to kAct state. In order to avoid overflow of 
// exp() function, sigmoid function is calculated based on the sign of 
// activation vector entry, as shown in the implementation below.
template <class T>
void SigmoidNode<T>::InitToAct() {
  // Sigmoid activation function:
  // f(z)=1.0/(1.0+exp(-z))
  for (size_t i = 0; i < activation_.rows(); ++i) {
    for (size_t j = 0; j < activation_.cols(); ++j) {
      T element_value = activation_(i, j);
      if (element_value >= 0.0) {
        activation_(i, j) = 1.0 / (1.0 + std::exp(-element_value));
      } else {
        activation_(i, j) = std::exp(element_value) / \
            (1.0 + std::exp(element_value));
      }
    }
  }
  current_act_state_ = kAct;  
}

template <class T>
void SigmoidNode<T>::ActToDropout() {
  activation_.array().unaryExpr(std::function<T(T)>( \
      BernoulliFunctor<T>(dropout_p_)));
  current_act_state_ = kDropout;
}

template <class T>
void SigmoidNode<T>::DropoutToPrime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  activation_.array() *= (1.0 - activation_.array());
  current_act_state_ = kPrime;
}

template <class T>
bool SigmoidNode<T>::Transition(ActStates state) {
  if (state == kFeed) {
    current_act_state_ = state;
    return true;
  }

  // Nothing happens if current node is an input node.
  // Note, an internal node changes to an input node permanently when 
  // Transition(kFeed) is called
  if (current_act_state_ == kFeed) {
    return true;
  }

  if (state == kInit) {
    current_act_state_ = state;
    return true;
  }

  if (current_act_state_ > state) {
    LOG(ERROR) << "Transition() for SigmoidNode is failed: "
               << "current state: " << current_act_state_
               << ", transition state: " << state;
    return false;
  }

  while (current_act_state_ < state) {
    if (!dropout_on_ && current_act_state_ == kAct) {
      current_act_state_ = kDropout;
    } 
    switch (current_act_state_) {
      case kInit: {
        InitToAct();
        break;
      }
      case kAct: {
        ActToDropout();
        break;
      }
      case kDropout: {
        DropoutToPrime();
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

template <class T>
void SigmoidNode<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& labels) {
  if (!Transition(kAct)) {
    LOG(ERROR) << "Evaluate() for SigmoidNode is failed.";
    exit(1);
  }

  CHECK_EQ(activation_.cols(), labels.cols())
      << "CalcLoss() for SigL2Node is failed: "
      << "activation and data matrix dimensions are not equal!";

  double accuracy = 0.0;
  size_t correct_guess = 0;

  if (activation_.rows() == 1) {
    correct_guess = (activation_.array().round() == \
        labels.array()).count();
  } else {
    for (size_t i = 0; i < labels.cols(); ++i) {
      size_t index_guess;
      activation_.col(i).maxCoeff(&index_guess);
      if (index_guess == labels(0, i)) {
        correct_guess++;
      }
    }
  }
  accuracy = correct_guess * 100.0 / labels.cols();
  std::cout << "Accuracy: " << accuracy << "%" << std::endl;
}

// Instantiate class, otherwise compilation will fail
template class SigmoidNode<float>;
template class SigmoidNode<double>;

}  // namespace intellgraph









