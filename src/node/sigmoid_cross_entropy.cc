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
#include "node/sigmoid_cross_entropy.h"

namespace intellgraph {

template <class T>
T SigCENode<T>::CalcLoss(const Eigen::Ref<const MatXX<T>>& labels) {
  T loss = 0;
  size_t batch_size = labels.cols();
  if (!Transition(kAct)) {
    LOG(ERROR) << "CalcLoss() for SigCENode is failed.";
    return -1.0;
  }
  CHECK_EQ(get_activation_ptr()->size(), labels.size())
      << "CalcLoss() for SigCENode is failed: "
      << "activation and data matrix dimensions are not equal!";

  loss = (labels.array() * get_activation_ptr()->array().log() + \
         (1.0 - labels.array()) * \
         (1.0 - get_activation_ptr()->array()).log()).sum();
  return -loss / batch_size;
}

template <class T>
bool SigCENode<T>::CalcDelta(const Eigen::Ref<const MatXX<T>>& labels) {
  if (!Transition(kAct)) {
    LOG(ERROR) << "CalcDelta() for SigCENode is failed.";
    return false;
  }

  CHECK_EQ(get_activation_ptr()->size(), labels.size())
      << "CalcDelta() for SigCENode is failed: "
      << "activation and data matrix dimensions are not equal!";

  get_delta_ptr()->array() = (get_activation_ptr()->array() \
      - labels.array());
  return true;
}

// Instantiate class, otherwise compilation will fail
template class SigCENode<float>;
template class SigCENode<double>;

}  // namespace intellgraph