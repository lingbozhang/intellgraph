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
	Huicheng Zhang <huichengz0520@gmail.com>
==============================================================================*/
#ifndef INTELLGRAPH_NODE_TANH_NODE_H_
#define INTELLGRAPH_NODE_TANH_NODE_H_

#include <functional>
#include <memory>

#include "glog/logging.h"
#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {
 
// TanhNode improves performance of CallActFxn and CalcActPrime with Eigen 
// library and has better performance than ActivationNode. 
template <class T>
class TanhNode : public Node<T> {
 public:
  TanhNode() = delete;

  explicit TanhNode(REF const NodeParameter& node_param)
      : Node<T>(node_param) {}

  // Move constructor
  TanhNode(MOVE TanhNode<T>&& rhs) = default;

  // Move operator
  TanhNode& operator=(MOVE TanhNode<T>&& rhs) = default;
  
  // Copy constructor and operator are explicitly deleted
  TanhNode(REF const TanhNode<T>& rhs) = delete;
  TanhNode& operator=(REF const TanhNode<T>& rhs) = delete;

  ~TanhNode() noexcept final = default;

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) final;

 protected:
  void Activate() final;

  void Prime() final;

};

// Alias for unqiue TanhNode pointer
template <class T>
using TanhNodeUPtr = std::unique_ptr<TanhNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_TANH_NODE_H_