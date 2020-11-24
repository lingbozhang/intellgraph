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
#ifndef INTELLGRAPH_SRC_VISITOR_RESIZE_VERTEX_VISITOR_H_
#define INTELLGRAPH_SRC_VISITOR_RESIZE_VERTEX_VISITOR_H_

#include "src/edge/dense_edge_impl.h"
#include "src/edge/op_vertex.h"
#include "src/visitor.h"

namespace intellgraph {

template <typename T> class ResizeVertexVisitor : public Visitor<T> {
public:
  ResizeVertexVisitor(int batch_size);
  ~ResizeVertexVisitor() override;

  void set_batch_size(int batch_size) { batch_size_ = batch_size; }
  void Visit(DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) override;

private:
  int batch_size_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class ResizeVertexVisitor<float>;
extern template class ResizeVertexVisitor<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_VISITOR_RESIZE_VERTEX_VISITOR_H_
