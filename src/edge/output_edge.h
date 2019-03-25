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
#ifndef INTELLGRAPH_EDGE_OUTPUT_EDGE_H_
#define INTELLGRAPH_EDGE_OUTPUT_EDGE_H_

#include "edge/edge.h"

#include "edge/edge_parameter.h"
#include "node/node.h"
#include "node/output_node.h"
#include "utility/common.h"

namespace intellgraph {
template <class T>
class OutputEdge : public Edge<T> {
 public:
  explicit OutputEdge(const struct EdgeParameter<T>& edge_param);

  ~OutputEdge() {}

  void PrintWeight() const final;

  void PrintNablaWeight() const final;

  void Forward() final;

  void Backward() final;

  // Passes a unary functor and applies it on the weight matrix
  void ApplyUnaryFunctor(std::function<T(T)> functor) final;

  MatXXSPtr<T> GetWeightPtr() final;

 private:
  const struct EdgeParameter<T> edge_param_;
  NodeSPtr<T> in_node_ptr_;
  NodeSPtr<T> out_node_ptr_;

  MatXXSPtr<T> weight_ptr_;
  MatXXSPtr<T> nabla_weight_ptr_;
};

// Alias for unique Edge pointer
template <class T>
using OutputEdgeSPtr = std::shared_ptr<OutputEdge<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_EDGE_OUTPUT_EDGE_H_







