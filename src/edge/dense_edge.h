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
#ifndef INTELLGRAPH_EDGE_DENSE_EDGE_H_
#define INTELLGRAPH_EDGE_DENSE_EDGE_H_

#include <functional>

#include "glog/logging.h"
#include "edge/edge.h"
#include "edge/edge_parameter.h"
#include "node/internal_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {
// DenseEdge is a edge class that used to build fully connected neural networks.
// In DenseEdge, weight is updated based on the backpropagation. 
template <class T>
class DenseEdge : implements Edge<T> {
 public:
  DenseEdge() noexcept = default;

  explicit DenseEdge(REF const EdgeParameter& edge_param);
  
  // Move constructor
  DenseEdge(MOVE DenseEdge<T>&& rhs) noexcept = default;

  // Move operator
  DenseEdge& operator=(MOVE DenseEdge<T>&& rhs) noexcept = default;

  // Copy constructor and operator are deleted
  DenseEdge(REF const DenseEdge<T>& rhs) = delete;
  DenseEdge& operator=(REF const DenseEdge<T>& rhs) = delete;

  ~DenseEdge() noexcept final = default;

  void PrintWeight() const final;

  void PrintNablaWeight() const final;

  virtual void Forward(MUTE IntNode<T>* node_in_ptr, \
                       MUTE IntNode<T>* node_out_ptr) final;

  virtual void Backward(MUTE IntNode<T>* node_in_ptr, \
                        MUTE IntNode<T>* node_out_ptr) final;

  void InitializeWeight(REF const std::function<T(T)>& functor) final;

  MUTE inline MatXX<T>* get_weight_ptr() const final {
    return weight_ptr_.get();
  }

  REF inline const MatXX<T>* ref_nabla_weight_ptr() const final {
    return nabla_weight_ptr_.get();
  }

 private:
  EdgeParameter edge_param_{};

  MatXXUPtr<T> weight_ptr_{nullptr};
  MatXXUPtr<T> nabla_weight_ptr_{nullptr};

};
// Alias for unique dense edge pointer
template <class T>
using DenseEdgeUPtr = std::unique_ptr<DenseEdge<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_EDGE_DENSE_EDGE_H_







