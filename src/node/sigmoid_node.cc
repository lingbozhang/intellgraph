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

// Transitions from initial state to activate state. In order to avoid overflow of
// exp() function, sigmoid function is calculated based on the sign of 
// activation vector entry, as shown in the implementation below.
template <class T>
void SigmoidNode<T>::Activate() {
  // Sigmoid activation function:
  // f(z)=1.0/(1.0+exp(-z))
  for (size_t i = 0; i < this->get_activation_ptr()->rows(); ++i) {
    for (size_t j = 0; j < this->get_activation_ptr()->cols(); ++j) {
      T element_value = this->get_activation_ptr()->array()(i, j);
      if (element_value >= 0.0) {
        this->get_activation_ptr()->array()(i, j) = 1.0 / (1.0 + \
            std::exp(-element_value));
      } else {
        this->get_activation_ptr()->array()(i, j) = std::exp(element_value) / \
            (1.0 + std::exp(element_value));
      }
    }
  }
}

template <class T>
void SigmoidNode<T>::Prime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  this->get_activation_ptr()->array() *= (1.0 - this->get_activation_ptr()->array());
}

template <class T>
void SigmoidNode<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& labels) {
  CHECK_EQ(this->get_activation_ptr()->cols(), labels.cols())
      << "CalcLoss() for SigL2Node is failed: "
      << "activation and data matrix dimensions are not equal!";

  double accuracy = 0.0;
  size_t correct_guess = 0;

  if (this->get_activation_ptr()->rows() == 1) {
    correct_guess = (this->get_activation_ptr()->array().round() == \
        labels.array()).count();
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

// Instantiate class, otherwise compilation will fail
template class SigmoidNode<float>;
template class SigmoidNode<double>;

}  // namespace intellgraph
