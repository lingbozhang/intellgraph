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

#include "edge/edge.h"
#include "edge/edge_parameter.h"
#include "utility/common.h"

namespace intellgraph {
// DenseEdge is a edge class that used to build fully connected neural networks.
// In DenseEdge, weight is updated based on the backpropagation. 
template <class T>
class DenseEdge : public Edge<T> {
 public:
  explicit DenseEdge(const EdgeParameter<T>& edge_param);

  ~DenseEdge() {}

  void PrintWeight() const final;

  void PrintNablaWeight() const final;

  virtual void Forward(NodeSPtr<T> node_in_ptr, NodeSPtr<T> node_out_ptr) final;

  virtual void Backward(NodeSPtr<T> node_in_ptr, NodeSPtr<T> node_out_ptr) final;

  void ApplyUnaryFunctor(std::function<T(T)> functor) final;

  MatXXSPtr<T> GetWeightPtr() final;

 private:
  const struct EdgeParameter<T> edge_param_;

  MatXXSPtr<T> weight_ptr_;
  MatXXSPtr<T> nabla_weight_ptr_;
};
// Alias for unique dense edge pointer
template <class T>
using DenseEdgeSPtr = std::shared_ptr<DenseEdge<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_EDGE_DENSE_EDGE_H_







