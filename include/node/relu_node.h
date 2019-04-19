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
#ifndef INTELLGRAPH_NODE_RELU_NODE_H_
#define INTELLGRAPH_NODE_RELU_NODE_H_ 

#include <functional>
#include <memory>

#include "glog/logging.h"
#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {
 
template <class T>
class ReLUNode : public Node<T> {
 public:
  ReLUNode() = delete;

  explicit ReLUNode(REF const NodeParameter& node_param)
      : Node<T>(node_param) {}

  // Move constructor
  ReLUNode(MOVE ReLUNode<T>&& rhs) = default;

  // Move operator
  ReLUNode& operator=(MOVE ReLUNode<T>&& rhs) = default;
  
  // Copy constructor and operator are explicitly deleted
  ReLUNode(REF const ReLUNode<T>& rhs) = delete;
  ReLUNode& operator=(REF const ReLUNode<T>& rhs) = delete;

  ~ReLUNode() noexcept final = default;

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) final;

 protected:
  void Activate() final;

  void Prime() final;

};

// Alias for unqiue ReLUNode pointer
//template <class T>
//using ReLUNodeUPtr = std::unique_ptr<ReLUNode<T>>;

}  // intellgraph

#endif  // INTELLGRAPH_NODE_RELU_NODE_H_