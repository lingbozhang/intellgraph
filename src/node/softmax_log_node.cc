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

// Transitions from kInit state to kAct state. 
template <class T>
void SoftmaxLogNode<T>::Activate() {
  this->get_activation_ptr()->array() = this->get_activation_ptr()->array().exp();
  VecX<T> vec = this->get_activation_ptr()->colwise().sum();
  this->get_activation_ptr()->array().rowwise() /= vec.transpose().array();
}

template <class T>
void SoftmaxLogNode<T>::Prime() {
  LOG(ERROR) << "DropoutToPrime() for SoftmaxLogNode is not defined";
  exit(1);
}

template <class T>
void SoftmaxLogNode<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& labels) {
  CHECK_EQ(this->get_activation_ptr()->cols(), labels.cols())
      << "CalcLoss() for SoftmaxLogNode is failed: "
      << "activation and data matrix dimensions are not equal!";

  double accuracy = 0.0;
  size_t correct_guess = 0;

  if (this->get_activation_ptr()->rows() == 1) {
    LOG(ERROR) << "Evaluate() for SoftmaxLogNode is failed";
    exit(1);
  } else {
    for (size_t i = 0; i < labels.cols(); ++i) {
      size_t index_guess;
      this->get_activation_ptr()->col(i).maxCoeff(&index_guess);
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

  CHECK_EQ(this->get_activation_ptr()->size(), labels.size())
    << "CalcLoss() for SoftmaxLogNode is failed: "
    << "activation and data matrix dimensions are not equal!";
  // log-like cost function 
  loss = -((this->get_activation_ptr()->array() * labels.array()). \
      colwise().maxCoeff().log()).sum();

  return loss / labels.rows();
}

template <class T>
bool SoftmaxLogNode<T>::CalcDelta(const Eigen::Ref<const MatXX<T>>& labels) {
  LOG(INFO) << "SoftmaxLogNode calculates delta";
  CHECK_EQ(this->get_activation_ptr()->size(), labels.size())
    << "CalcDelta() for SoftmaxLogNode is failed: "
    << "activation and data matrix dimensions are not equal!";

  this->get_delta_ptr()->array() = (this->get_activation_ptr()->array() \
    - labels.array());
  return true;
}

// Instantiate class, otherwise compilation will fail
template class SoftmaxLogNode<float>;
template class SoftmaxLogNode<double>;

}  // namespace intellgraph









