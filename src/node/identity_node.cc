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

// Transitions from kInit state to kAct state.
template <class T>
void IdentityNode<T>::Activate() {
  // Identity activation function
}

template <class T>
void IdentityNode<T>::Prime() {
  // Derivative equation
  this->get_activation_ptr()->array() = 1.0;
}

template <class T>
void IdentityNode<T>::Evaluate(const Eigen::Ref<const MatXX<T>>& labels) {
  CHECK_EQ(this->get_activation_ptr()->cols(), labels.cols())
      << "CalcLoss() for IdentityNode is failed: "
      << "activation and data matrix dimensions are not equal!";

  T loss = (this->get_activation_ptr()->array() - labels.array()). \
            matrix().norm();
  T avg_norm = loss / labels.cols();
  std::cout << "Average l2 norm: " << avg_norm << std::endl;
}

// Instantiate class, otherwise compilation will fail
template class IdentityNode<float>;
template class IdentityNode<double>;

}  // namespace intellgraph









