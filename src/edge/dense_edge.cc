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
DenseEdge<T>::DenseEdge(const struct EdgeParameter<T>& edge_param)
    : edge_param_(edge_param) {
  in_node_ptr_ = edge_param.in_node_ptr;
  out_node_ptr_ = edge_param.out_node_ptr;

  std::vector<size_t> in_dims = in_node_ptr_->GetDims();
  std::vector<size_t> out_dims = out_node_ptr_->GetDims();

  weight_ptr_ = std::make_unique<MatXX<T>>(in_dims[0], out_dims[0]);
  nabla_weight_ptr_ = std::make_unique<MatXX<T>>(in_dims[0], out_dims[0]);
  
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
void DenseEdge<T>::Forward() {
  MatXXSPtr<T> weighted_sum_ptr = std::make_shared<MatXX<T>>(
      weight_ptr_->transpose() * in_node_ptr_->GetActivationPtr()->matrix() + \
      out_node_ptr_->GetBiasPtr()->matrix());
  
  out_node_ptr_->SetActivationPtr(weighted_sum_ptr);
}

template <class T>
void DenseEdge<T>::Backward() {
  // $\nabla W^{l}=a^{l-1}(\delta^{l})^T$
  nabla_weight_ptr_->matrix() = in_node_ptr_->GetActivationPtr()->matrix() * \
                                out_node_ptr_->GetDeltaPtr()->transpose();
  // Calculates delta_ptr_ of input node
  // $\delta^l= \mathcal{D}[f^\prime(z^l)]W^{l+1}\delta^{l+1}$
  
  MatXXSPtr<T> in_delta_ptr = std::make_shared<MatXX<T>>(
      weight_ptr_->matrix() * out_node_ptr_->GetDeltaPtr()->matrix());

  in_node_ptr_->CalcActPrime();
  in_delta_ptr->array() *= in_node_ptr_->GetActivationPtr()->array();

  in_node_ptr_->SetDeltaPtr(in_delta_ptr);
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

