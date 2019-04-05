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
#include "node/act_loss_node.h"

namespace intellgraph {

template <class T>
T ActLossNode<T>::CalcLoss(const MatXX<T>* data_result_ptr) {
  T loss = 0;
  if (!Transition(kAct)) {
    std::cout << "ERROR: CalcLoss() for ActLossNode fails. " 
              << "Transition to kAct fails" << std::endl;
    exit(1);
  }
  auto loss_functor = ref_node_param().ref_loss_functor();
  if (loss_functor == nullptr) {
    std::cout << "WARNING: loss function is not defined." << std::endl;
  } else {
    loss = loss_functor(get_activation_ptr(), data_result_ptr);
  }
  return loss;
}

template <class T>
void ActLossNode<T>::CalcDelta(const MatXX<T>* data_result_ptr) {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CalcDelta() for ActLossNode fails. " 
              << "Transition to kAct fails" << std::endl;
    exit(1);
  }
  auto loss_prime_functor = ref_node_param().ref_loss_prime_functor();
  if (loss_prime_functor == nullptr) {
    std::cout << "WARNING: loss prime function is not defined." << std::endl;
  } else {
    loss_prime_functor(get_activation_ptr(), data_result_ptr, get_delta_ptr());
  }
  // Note CalcActPrime overwrites data in activation_ptr in-place
  if (!Transition(kPrime)) {
    std::cout << "ERROR: CalcDelta() for ActLossNode fails. " 
              << "Transition to kPrime fails" << std::endl;
    exit(1);
  }
  get_delta_ptr()->array() *= get_activation_ptr()->array();
}

// Instantiate class, otherwise compilation will fail
template class ActLossNode<float>;
template class ActLossNode<double>;

}  // namespace intellgraph