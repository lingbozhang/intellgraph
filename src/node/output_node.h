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
#ifndef INTELLGRAPH_NODE_OUTPUT_NODE_H_
#define INTELLGRAPH_NODE_OUTPUT_NODE_H_

#include <functional>
#include <memory>
#include <vector>

#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

template <class T>
class OutputNode : public Node<T> {
 public:
  OutputNode() noexcept {}

  explicit OutputNode(REF const NodeParameter& node_param)
      : Node<T>(node_param) {}

  // Move constructor
  OutputNode(MOVE OutputNode<T>&& rhs) = default;

  // Move operator
  OutputNode& operator=(MOVE OutputNode<T>&& rhs) = default;

  // Copy constructor and operator are explicitly deleted
  OutputNode(REF const OutputNode<T>& rhs) = delete;
  OutputNode& operator=(REF const OutputNode<T>& rhs) = delete;

  virtual ~OutputNode() noexcept = default;

  COPY virtual T CalcLoss(REF const Eigen::Ref<const MatXX<T>>& labels) = 0;

  virtual bool CalcDelta(REF const Eigen::Ref<const MatXX<T>>& labels) = 0;

};

// Alias for unique node pointer
template <class T>
using OutputNodeUPtr = std::unique_ptr<OutputNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_OUTPUT_NODE_H_







  