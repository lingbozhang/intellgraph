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
DenseEdge<T>::DenseEdge(const EdgeParameter& edge_param) {
  edge_param_.Clone(edge_param);

  size_t row = edge_param.ref_dims_in()[0];
  size_t col = edge_param.ref_dims_out()[0];

  weight_ptr_ = std::make_unique<MatXX<T>>(row, col);
  nabla_weight_ptr_ = std::make_unique<MatXX<T>>(row, col);
  
  weight_ptr_->array() = 0.0;
  nabla_weight_ptr_->array() = 0.0;
}

template <class T>
void DenseEdge<T>::PrintWeight() const {
  std::cout << "DenseEdge: " << edge_param_.ref_id() << " Weight matrix:"
            << std::endl << weight_ptr_->array() << std::endl;
}

template <class T>
void DenseEdge<T>::PrintNablaWeight() const {
  std::cout << "DenseEdge: " << edge_param_.ref_id() << " Nabla weight matrix:"
            << std::endl << nabla_weight_ptr_->array() << std::endl;    
}

template <class T>
void DenseEdge<T>::Forward(const IntNode<T>* node_in_ptr, \
    IntNode<T>* node_out_ptr) {
  CHECK_EQ(weight_ptr_->rows(), node_in_ptr->get_activation_ptr()->rows())
      << "Forward() in DenseEdge is failed:"
      << "Dimensions of weight and activation from input node are not equal.";
  
  CHECK_EQ(weight_ptr_->cols(), node_out_ptr->get_activation_ptr()->rows())
      << "Forward() in DenseEdge is failed:"
      << "Dimensions of weight and activation from output node are not equal.";

  node_out_ptr->get_activation_ptr()->matrix().noalias() = \
      (weight_ptr_->transpose() * \
      node_in_ptr->get_activation_ptr()->matrix()).colwise() + \
      *node_out_ptr->get_bias_ptr();
  // Updates the activation matrix state
  node_out_ptr->ResetActState();
}

template <class T>
void DenseEdge<T>::Backward(IntNode<T>* node_in_ptr, \
    IntNode<T>* node_out_ptr) {
  // $\nabla W^{l}=a^{l-1}(\delta^{l})^T$
  size_t batch_size = node_in_ptr->get_activation_ptr()->cols();
  nabla_weight_ptr_->matrix().noalias() = \
      (node_in_ptr->get_activation_ptr()->matrix() * \
      node_out_ptr->get_delta_ptr()->transpose()) / batch_size;
  // Calculates delta_ptr_ of input node
  // $\delta^l= \mathcal{D}[f^\prime(z^l)]W^{l+1}\delta^{l+1}$
  node_in_ptr->get_delta_ptr()->matrix().noalias() = \
      weight_ptr_->matrix() * node_out_ptr->get_delta_ptr()->matrix();

  node_in_ptr->CalcActPrime();

  node_in_ptr->get_delta_ptr()->array() *= \
      node_in_ptr->get_activation_ptr()->array();
}

template <class T>
void DenseEdge<T>::InitializeWeight(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    LOG(WARNING) << "functor passed to InitializeWeight() is not defined: "
                 << "Initializes weight with standard normal distribution.";
    weight_ptr_->array() = weight_ptr_->array().unaryExpr( \
        std::function<T(T)>(NormalFunctor<T>(0.0, 1.0)));
  } else {
    weight_ptr_->array() = weight_ptr_->array().unaryExpr(functor);
  }
}

// Instantiate class, otherwise compilation will fail
template class DenseEdge<float>;
template class DenseEdge<double>;

} // namespace intellgraph

