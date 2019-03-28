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
DenseEdge<T>::DenseEdge(const EdgeParameter<T>& edge_param)
    : edge_param_(edge_param) {
  size_t dims_in = edge_param.dims_in[0];
  size_t dims_out = edge_param.dims_out[0];

  weight_ptr_ = std::make_shared<MatXX<T>>(dims_in, dims_out);
  nabla_weight_ptr_ = std::make_shared<MatXX<T>>(dims_in, dims_out);
  
  weight_ptr_->array() = 0.0;
  nabla_weight_ptr_->array() = 0.0;
}

template <class T>
void DenseEdge<T>::PrintWeight() const {
  std::cout << "DenseEdge " << edge_param_.id << " Weight matrix:"
            << std::endl << weight_ptr_->array() << std::endl;
}

template <class T>
void DenseEdge<T>::PrintNablaWeight() const {
  std::cout << "DenseEdge " << edge_param_.id << " Nabla weight matrix:"
            << std::endl << nabla_weight_ptr_->array() << std::endl;    
}

template <class T>
void DenseEdge<T>::Forward(NodeSPtr<T> node_in_ptr, NodeSPtr<T> node_out_ptr) {
  MatXXSPtr<T> weighted_sum_ptr = std::make_shared<MatXX<T>>(
      weight_ptr_->transpose() * node_in_ptr->GetActivationPtr()->matrix() + \
      node_out_ptr->GetBiasPtr()->matrix());
  
  node_out_ptr->SetActivationPtr(weighted_sum_ptr);
}

template <class T>
void DenseEdge<T>::Backward(NodeSPtr<T> node_in_ptr, NodeSPtr<T> node_out_ptr) {
  // $\nabla W^{l}=a^{l-1}(\delta^{l})^T$
  nabla_weight_ptr_->matrix() = node_in_ptr->GetActivationPtr()->matrix() * \
                                node_out_ptr->GetDeltaPtr()->transpose();
  // Calculates delta_ptr_ of input node
  // $\delta^l= \mathcal{D}[f^\prime(z^l)]W^{l+1}\delta^{l+1}$
  
  MatXXSPtr<T> delta_in_ptr = std::make_shared<MatXX<T>>(
      weight_ptr_->matrix() * node_out_ptr->GetDeltaPtr()->matrix());

  node_in_ptr->CalcActPrime();
  delta_in_ptr->array() *= node_in_ptr->GetActivationPtr()->array();

  node_in_ptr->SetDeltaPtr(delta_in_ptr);
}

template <class T>
void DenseEdge<T>::ApplyUnaryFunctor(std::function<T(T)> functor) {
  if (functor == nullptr) {
    std::cout << "WARNING: functor passed to ApplyUnaryFunctor() is not defined." 
              << std::endl;
  } else {
    weight_ptr_->array() = weight_ptr_->array().unaryExpr(functor);
  }
}

template <class T>
MatXXSPtr<T> DenseEdge<T>::GetWeightPtr() {
  return weight_ptr_;
}

// Instantiate class, otherwise compilation will fail
template class DenseEdge<float>;
template class DenseEdge<double>;

} // namespace intellgraph

