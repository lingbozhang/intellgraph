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
#include "node/softmax_log_node.h"

namespace intellgraph {

template <class T>
SoftmaxLogNode<T>::SoftmaxLogNode(const NodeParameter& node_param) {
  node_param_.Clone(node_param);

  size_t row = node_param.ref_dims()[0];
  size_t col = node_param.ref_dims()[1];

  bias_ptr_ = std::make_unique<VecX<T>>(row);

  bias_ptr_->array() = 0.0;

  current_act_state_ = kInit;
}

template <class T>
void SoftmaxLogNode<T>::PrintBias() const {
  std::cout << "Node: " << node_param_.ref_id() << std::endl 
            << "Bias Vector:" << std::endl << bias_ptr_->array() 
            << std::endl;
}

template <class T>
void SoftmaxLogNode<T>::InitializeBias(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    LOG(WARNING) << "InitializeBias() for SoftmaxLogNode is failed: " 
                 << "initializes bias with standard normal distribution";
    bias_ptr_->array() = bias_ptr_->array().unaryExpr(std::function<T(T)>( \
        NormalFunctor<T>(0.0, 1.0)));
  } else {
    bias_ptr_->array() = bias_ptr_->array().unaryExpr(functor);
  }
  Transition(kInit);
}

// Transitions from kInit state to kAct state. 
template <class T>
void SoftmaxLogNode<T>::InitToAct() {
  activation_.array() = activation_.array().exp();
  VecX<T> vec = activation_.colwise().sum();
  activation_.array().rowwise() /= vec.transpose().array();
  current_act_state_ = kAct;  
}

template <class T>
void SoftmaxLogNode<T>::ActToPrime() {
  LOG(ERROR) << "ActToPrime() for SoftmaxLogNode is node defined";
  exit(1);
}

template <class T>
bool SoftmaxLogNode<T>::Transition(ActStates state) {
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
    LOG(ERROR) << "Transition() for SoftmaxLogNode is failed: "
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
        LOG(ERROR) << "Transition() for SoftmaxLogNode is failed";
        return false;
      }
    }
  }
  return true;
}

template <class T>
bool SoftmaxLogNode<T>::CallActFxn() {
  if (!Transition(kAct)) {
    LOG(ERROR) << "CallActFxn() for SoftmaxLogNode is failed";
    return false;
  }
  return true;
}

template <class T>
bool SoftmaxLogNode<T>::CalcActPrime() {
  if (!Transition(kPrime)) {
    LOG(ERROR) << "CalcActPrime() for SoftmaxLogNode is failed";
    return false;
  }
  return true;
}

template <class T>
void SoftmaxLogNode<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& labels) {
  if (!Transition(kAct)) {
    LOG(ERROR) << "Evaluate() for SoftmaxLogNode is failed.";
    exit(1);
  }

  CHECK_EQ(activation_.cols(), labels.cols())
      << "CalcLoss() for SoftmaxLogNode is failed: "
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

template <class T>
T SoftmaxLogNode<T>::CalcLoss(const Eigen::Ref<const MatXX<T>>& labels) {
  T loss = 0;
  size_t batch_size = labels.cols();
  if (!Transition(kAct)) {
    LOG(ERROR) << "CalcLoss() for SoftmaxLogNode is failed.";
    return -1.0;
  }
  CHECK_EQ(get_activation_ptr()->size(), labels.size())
    << "CalcLoss() for SoftmaxLogNode is failed: "
    << "activation and data matrix dimensions are not equal!";
  // log-like cost function 
  loss = -((activation_.array() * labels.array()). \
      colwise().maxCoeff().log()).sum();

  return loss / labels.rows();
}

template <class T>
bool SoftmaxLogNode<T>::CalcDelta(const Eigen::Ref<const MatXX<T>>& labels) {
  if (!Transition(kAct)) {
    LOG(ERROR) << "CalcDelta() for SoftmaxLogNode is failed.";
    return false;
  }

  CHECK_EQ(get_activation_ptr()->size(), labels.size())
    << "CalcDelta() for SoftmaxLogNode is failed: "
    << "activation and data matrix dimensions are not equal!";

  get_delta_ptr()->array() = (get_activation_ptr()->array() \
    - labels.array());
  return true;
}

// Instantiate class, otherwise compilation will fail
template class SoftmaxLogNode<float>;
template class SoftmaxLogNode<double>;

}  // namespace intellgraph









