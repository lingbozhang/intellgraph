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
#include "edge/dense_edge.h"

namespace intellgraph {

template <class T>
void DenseEdge<T>::Forward(Node<T>* node_in_ptr, \
    Node<T>* node_out_ptr) {
  CHECK_EQ(this->get_weight_ptr()->rows(), node_in_ptr->get_activation_ptr()->rows())
      << "Forward() in DenseEdge is failed:"
      << "Dimensions of weight and activation from input node are not equal.";

  CHECK_EQ(this->get_weight_ptr()->cols(), node_out_ptr->get_activation_ptr()->rows())
      << "Forward() in DenseEdge is failed:"
      << "Dimensions of weight and activation from output node are not equal.";

  // Note activation matrix value is added rather than overwritten
  node_out_ptr->get_activation_ptr()->noalias() += \
      (this->get_weight_ptr()->transpose() * \
      node_in_ptr->get_activation_ptr()->matrix()).colwise() + \
      *node_out_ptr->get_bias_ptr();
  // Updates the activation matrix state
  node_out_ptr->ToInit();
}

template <class T>
void DenseEdge<T>::Backward(Node<T>* node_in_ptr, \
    Node<T>* node_out_ptr) {
  // $\nabla W^{l}=a^{l-1}(\delta^{l})^T$
  size_t batch_size = node_in_ptr->get_activation_ptr()->cols();
  this->get_nabla_weight_ptr()->matrix().noalias() = \
      (node_in_ptr->get_activation_ptr()->matrix() * \
      node_out_ptr->get_delta_ptr()->transpose()) / batch_size;
  // Calculates delta_ptr_ of input node
  // $\delta^l= \mathcal{D}[f^\prime(z^l)]W^{l+1}\delta^{l+1}$
  CHECK_EQ(this->get_weight_ptr()->cols(), node_out_ptr->get_activation_ptr()->rows())
      << "Backward() in DenseEdge is failed:"
      << "Dimensions of weight and activation from output node are not equal.";

  node_in_ptr->get_delta_ptr()->noalias() = \
      this->get_weight_ptr()->matrix() * node_out_ptr->get_delta_ptr()->matrix();

  node_in_ptr->ToPrime();

  node_in_ptr->get_delta_ptr()->array() *= \
      node_in_ptr->get_activation_ptr()->array();
}

// Instantiate class, otherwise compilation will fail
template class DenseEdge<float>;
template class DenseEdge<double>;

} // namespace intellgraph

