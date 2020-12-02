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
#ifndef INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_
#define INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_

#include "src/graph.h"

namespace intellgraph {

template <typename T> class RnnImpl : public Graph {
public:
  RnnImpl() = default;
  ~RnnImpl() = default;

  void Train();

private:
  // Graph topology
  const typename Graph::AdjacencyList adjacency_list_;
  std::vector<int> topological_order_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_RNN_IMPL_H_
