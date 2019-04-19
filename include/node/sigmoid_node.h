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
#ifndef INTELLGRAPH_NODE_SIGMOID_NODE_H_
#define INTELLGRAPH_NODE_SIGMOID_NODE_H_

#include <functional>
#include <memory>

#include "glog/logging.h"
#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {
 
// Forward declaration
template <class T>
class SigL2Node;

template <class T>
class SigCENode;

// SigmoidNode improves performance of CallActFxn and CalcActPrime with Eigen 
// library and has better performance than ActivationNode. 
template <class T>
class SigmoidNode : public Node<T> {
 public:
  friend class SigL2Node<T>;

  friend class SigCENode<T>;

  SigmoidNode() = delete;

  explicit SigmoidNode(REF const NodeParameter& node_param)
      : Node<T>(node_param) {}

  // Move constructor
  SigmoidNode(MOVE SigmoidNode<T>&& rhs) = default;

  // Move operator
  SigmoidNode& operator=(MOVE SigmoidNode<T>&& rhs) = default;
  
  // Copy constructor and operator are explicitly deleted
  SigmoidNode(REF const SigmoidNode<T>& rhs) = delete;
  SigmoidNode& operator=(REF const SigmoidNode<T>& rhs) = delete;

  ~SigmoidNode() noexcept final = default;

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) final;

 protected:
  void Activate() final;

  void Prime() final;

};

// Alias for unqiue SigmoidNode pointer
template <class T>
using SigNodeUPtr = std::unique_ptr<SigmoidNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_SIGMOID_NODE_H_







