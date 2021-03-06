/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-1.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
        Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#ifndef INTELLGRAPH_SRC_EDGE_EDGE_H_
#define INTELLGRAPH_SRC_EDGE_EDGE_H_

#include "src/eigen.h"
#include "src/solver.h"
#include "src/visitor.h"

namespace intellgraph {

// The Edge class is an abstract class that represents an edge data in the
// IntellGraph. The class is used to store weight and nabla weight matrices from
// the Neural Network, and it adopts the Visitor Pattern, and accepts various
// visitors that apply different operations on the edge data members
template <typename T> class Edge {
public:
  Edge() = default;
  virtual ~Edge() = default;

  virtual void Accept(Visitor<T> &visitor) = 0;
  virtual void Accept(Solver<T> &solver) = 0;

  virtual int id() const = 0;
  virtual int row() const = 0;
  virtual int col() const = 0;
  virtual const Eigen::Map<const MatrixX<T>> &weight() = 0;
  virtual Eigen::Map<MatrixX<T>> mutable_weight() = 0;
  virtual Eigen::Map<MatrixX<T>> mutable_bias() = 0;
  virtual Eigen::Map<MatrixX<T>> mutable_weight_stores(int index) = 0;
  virtual Eigen::Map<MatrixX<T>> mutable_bias_stores(int index) = 0;

  virtual const MatrixX<T> CalcNablaWeight() = 0;
  virtual const MatrixX<T> CalcNablaBias() = 0;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_EDGE_H_
