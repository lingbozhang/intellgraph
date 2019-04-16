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
#include "node/identity_l2_node.h"

namespace intellgraph {

template <class T>
T IDL2Node<T>::CalcLoss(const Eigen::Ref<const MatXX<T>>& labels) {
  T loss = 0;
  size_t batch_size = labels.cols();

  CHECK_EQ(this->get_activation_ptr()->size(), labels.size())
      << "CalcLoss() for IDL2Node is failed: "
      << "activation and data matrix dimensions are not equal!";

  loss = (this->get_activation_ptr()->array() - labels.array()). \
          matrix().squaredNorm();
  return loss / 2.0 / batch_size;
}

template <class T>
bool IDL2Node<T>::CalcDelta(const Eigen::Ref<const MatXX<T>>& labels) {
  LOG(INFO) << "IDL2Node calculates delta";
  CHECK_EQ(this->get_activation_ptr()->size(), labels.size())
      << "CalcDelta() for IDL2Node is failed: "
      << "activation and data matrix dimensions are not equal!";

  this->get_delta_ptr()->array() = (this->get_activation_ptr()->array() \
      - labels.array());

  this->ToPrime();

  this->get_delta_ptr()->array() *= this->get_activation_ptr()->array();
  return true;
}

// Instantiate class, otherwise compilation will fail
template class IDL2Node<float>;
template class IDL2Node<double>;

}  // namespace intellgraph