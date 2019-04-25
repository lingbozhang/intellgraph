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
  Huicheng Zhang <huichengz0520@gmail.com>
==============================================================================*/
#include "node/tanh_node.h"

namespace intellgraph {

// Transitions from initial state to activate state. In order to avoid overflow of
// exp() function, tanh function is calculated based on the sign of 
// activation vector entry, as shown in the implementation below.
template <class T>
void TanhNode<T>::Activate() {
  // Tanh activation function:
  this->get_activation_ptr()->array() = \
      this->get_activation_ptr()->array().tanh();
}

template <class T>
void TanhNode<T>::Prime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  this->get_activation_ptr()->array() = 1.0 - \
      this->get_activation_ptr()->array().pow(2);
}

template <class T>
void TanhNode<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& labels) {
  CHECK_EQ(this->get_activation_ptr()->cols(), labels.cols())
      << "CalcLoss() for SigL2Node is failed: "
      << "activation and data matrix dimensions are not equal!";

  double accuracy = 0.0;
  size_t correct_guess = 0;

  if (this->get_activation_ptr()->rows() == 1) {
    correct_guess = (this->get_activation_ptr()->array().round() == \
        labels.array()).count();
  } else {
    LOG(ERROR) << "Evaluate() for TanhNode is not defined";
  }
  accuracy = correct_guess * 100.0 / labels.cols();
  std::cout << "Accuracy: " << accuracy << "%" << std::endl;
}

// Instantiate class, otherwise compilation will fail
template class TanhNode<float>;
template class TanhNode<double>;

}  // namespace intellgraph
