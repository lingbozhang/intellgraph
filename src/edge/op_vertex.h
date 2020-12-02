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
#ifndef INTELLGRAPH_SRC_EDGE_VERTEX_OP_VERTEX_H_
#define INTELLGRAPH_SRC_EDGE_VERTEX_OP_VERTEX_H_

#include "src/eigen.h"

namespace intellgraph {

// OpVertex is an abstract class that represents a vertex in the IntellGraph.
// The class is used to store activation and bias matrices in the Neural
// Network.
template <typename T> class OpVertex {
public:
  typedef T value_type;

  OpVertex() = default;
  virtual ~OpVertex() = default;

  virtual void Activate() = 0;
  virtual void Derive() = 0;
  // Resizes activation and delta matrices
  virtual void ResizeVertex(int length) = 0;

  virtual int id() const = 0;
  virtual int row() const = 0;
  virtual int col() const = 0;

  virtual const MatrixX<T> &activation() const = 0;
  virtual MatrixX<T> *mutable_activation() = 0;

  virtual MatrixX<T> *mutable_delta() = 0;
  virtual VectorX<T> *mutable_bias() = 0;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_VERTEX_OP_VERTEX_H_
