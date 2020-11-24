/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
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
#ifndef INTELLGRAPH_SRC_VISITOR_H_
#define INTELLGRAPH_SRC_VISITOR_H_

#include "src/edge/op_vertex.h"

namespace intellgraph {

// Forward declaration
template <typename T, class V1, class V2> class DenseEdgeImpl;

template <typename T> class Visitor {
public:
  Visitor() = default;
  virtual ~Visitor() = default;

  virtual void Visit(DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) = 0;
};

template class Visitor<float>;
template class Visitor<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_VISITOR_H_
