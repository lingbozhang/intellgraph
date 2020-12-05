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
#ifndef INTELLGRAPH_SRC_GRAPH_CLASSIFIER_BUILDER_H_
#define INTELLGRAPH_SRC_GRAPH_CLASSIFIER_BUILDER_H_

#include <string>

#include "src/graph/classifier_impl.h"
#include "src/graph/graph_builder.h"

namespace intellgraph {

template <typename T> class ClassifierBuilder {
public:
  ClassifierBuilder();
  ~ClassifierBuilder();

  ClassifierBuilder &AddEdge(const std::string &edge_type,
                             const VertexParameter &vtx_param_in,
                             const VertexParameter &vtx_param_out);
  ClassifierBuilder &SetInputVertexId(int id);
  ClassifierBuilder &SetOutputVertexId(int id);
  ClassifierBuilder &SetInitVisitor(std::unique_ptr<Visitor<T>> init_visitor);
  ClassifierBuilder &SetSolver(std::unique_ptr<Solver<T>> solver);
  std::unique_ptr<ClassifierImpl<T>> Build();

private:
  GraphBuilder<T> graph_builder_;
  std::unique_ptr<Visitor<T>> init_visitor_;
  std::unique_ptr<Solver<T>> solver_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class ClassifierBuilder<float>;
extern template class ClassifierBuilder<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_CLASSIFIER_BUILDER_H_
