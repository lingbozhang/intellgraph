/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 3.0 (the "License");
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
#ifndef INTELLGRAPH_SRC_SOLVER_ADA_MAX_H_
#define INTELLGRAPH_SRC_SOLVER_ADA_MAX_H_

#include "src/edge.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/solver.h"

namespace intellgraph {

template <typename T> class AdaMax : public Solver<T> {
public:
  explicit AdaMax(T eta = 0.002, T lambda = 0.0, T beta1 = 0.9, T beta2 = 0.999);
  ~AdaMax() override;

  void Visit(Edge<T> &edge) override;

private:
  T eta_ = 0;
  T lambda_ = 0;
  T beta1_ = 0;
  T beta2_ = 0;
  int iteration_count_ = 0;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class AdaMax<float>;
extern template class AdaMax<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_SOLVER_ADA_MAX_H_
