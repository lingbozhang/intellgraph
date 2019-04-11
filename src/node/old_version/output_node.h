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

#include <memory>

#include "node/node.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// OutputNode interface has two member methods:
// 1. Calculates value of loss function
// 2. Calculates delta $dl/dz$
template <class T>
interface OutputNode : implements Node<T> {
 public:
  virtual ~OutputNode() noexcept = default;

  COPY virtual T CalcLoss(REF const MatXX<T>* result_data_ptr) = 0;

  virtual bool CalcDelta(REF const MatXX<T>* result_data_ptr) = 0;

};

// Alias for unique node pointer
template <class T>
using OutputNodeUPtr = std::unique_ptr<OutputNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_OUTPUT_NODE_H_







  