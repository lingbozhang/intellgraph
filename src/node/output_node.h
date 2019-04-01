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
#include "utility/common.h"

namespace intellgraph {
// OutputNode is an abstract class and contains two methods:
// 1. GetLoss: calculates prediction loss of the neural networks
// 2. CalcDelta: calculates $dloss/dz$
template <class T>
class OutputNode : public Node<T> {
 public:
  OutputNode() noexcept = default;
  
  virtual ~OutputNode() noexcept = default;
  
  // Function name with letter 'k' indicates a reference function (which means
  // arguments are referred in the function)
  virtual T CalcLoss_k(const MatXX<T>& data_result) = 0;

  virtual void CalcDelta_k(const MatXX<T>& data_result) = 0;

};

// Alias for shared node pointer
template <class T>
using OutputNodeUPtr = std::unique_ptr<OutputNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_OUTPUT_NODE_H_







  