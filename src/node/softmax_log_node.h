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
#ifndef INTELLGRAPH_NODE_SOFTMAX_LOG_NODE_H_
#define INTELLGRAPH_NODE_SOFTMAX_LOG_NODE_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "node/node_parameter.h"
#include "node/output_node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {

template<class T>
class SoftmaxLogNode : public OutputNode<T> {
 public:
  SoftmaxLogNode() noexcept = delete;

  explicit SoftmaxLogNode(REF const NodeParameter& node_param)
      : OutputNode<T>(node_param) {}

  // Move constructor
  SoftmaxLogNode(MOVE SoftmaxLogNode<T>&& rhs) = default;

  // Move operator
  SoftmaxLogNode& operator=(MOVE SoftmaxLogNode<T>&& rhs) = default;

  // Copy constructor and operator are explicitly deleted
  SoftmaxLogNode(REF const SoftmaxLogNode<T>& rhs) = delete;
  SoftmaxLogNode& operator=(REF const SoftmaxLogNode<T>& rhs) = delete;

  ~SoftmaxLogNode() noexcept final = default;

  void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) final;

  COPY T CalcLoss(REF const Eigen::Ref<const MatXX<T>>& labels) final;

  bool CalcDelta(REF const Eigen::Ref<const MatXX<T>>& labels) final;

 protected:
  void Activate() final;

  void Prime() final;

};

// Alias for unique SoftmaxLogNode pointer
template <class T>
using SMLogNodeUPtr = std::unique_ptr<SoftmaxLogNode<T>>;
 
}  // namespace intellgraph
 
# endif  // INTELLGRAPH_NODE_SOFTMAX_LOG_NODE_H_
 