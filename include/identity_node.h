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
#ifndef INTELLGRAPH_NODE_IDENTITY_NODE_H_
#define INTELLGRAPH_NODE_IDENTITY_NODE_H_

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
class IDL2Node;

template <class T>
class IdentityNode : public Node<T> {
 public:
  friend class IDL2Node<T>;

  IdentityNode() = delete;

  explicit IdentityNode(REF const NodeParameter& node_param)
      : Node<T>(node_param) {}

  // Move constructor
  IdentityNode(MOVE IdentityNode<T>&& rhs) = default;

  // Move operator
  IdentityNode& operator=(MOVE IdentityNode<T>&& rhs) noexcept = default;
  
  // Copy constructor and operator are explicitly deleted
  IdentityNode(REF const IdentityNode<T>& rhs) = delete;
  IdentityNode& operator=(REF const IdentityNode<T>& rhs) = delete;

  virtual ~IdentityNode() noexcept final = default;

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) final;

 private:
  void Activate() final;

  void Prime() final;

};

// Alias for unqiue IdentityNode pointer
template <class T>
using IDNodeUPtr = std::unique_ptr<IdentityNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_IDENTITY_NODE_H_







