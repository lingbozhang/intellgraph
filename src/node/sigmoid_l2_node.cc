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
SigL2Node<T>::SigL2Node(REF const NodeParameter<T>& node_param) {
  NodeParameter<T> node_param_new;
  node_param_new.Clone(node_param);
  node_param_new.move_node_name("SigmoidNode");
  node_ptr_ = std::make_unique<SigmoidNode<T>>(node_param_new);
}

template <class T>
T SigL2Node<T>::CalcLoss(const MatXX<T>* data_result_ptr) {
  T loss = 0;
  if (!Transition(kAct)) {
    LOG(ERROR) << "CalcLoss() for SigL2Node is failed.";
    return -1.0;
  }
  CHECK_EQ(get_activation_ptr()->size(), data_result_ptr->size()) 
      << "CalcLoss() for SigL2Node is failed: "
      << "activation and data matrix dimensions are not equal!";

  loss = (get_activation_ptr()->array() - data_result_ptr->array()). \
          matrix().squaredNorm();
  return loss;
}

template <class T>
bool SigL2Node<T>::CalcDelta(const MatXX<T>* data_result_ptr) {
  if (!Transition(kAct)) {
    LOG(ERROR) << "CalcDelta() for SigL2Node is failed.";
    return false;
  }

  CHECK_EQ(get_activation_ptr()->size(), data_result_ptr->size()) 
      << "CalcDelta() for SigL2Node is failed: "
      << "activation and data matrix dimensions are not equal!";

  get_delta_ptr()->array() = 2.0 * (get_activation_ptr()->array() \
      - data_result_ptr->array());
  // Note CalcActPrime overwrites data in activation_ptr_ in-place
  if (!Transition(kPrime)) {
    LOG(ERROR) << "CalcDelta() for SigL2Node is failed.";
    return false;
  }
  get_delta_ptr()->array() *= get_activation_ptr()->array();
  return true;
}

// Instantiate class, otherwise compilation will fail
template class SigL2Node<float>;
template class SigL2Node<double>;

}  // namespace intellgraph