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
#ifndef INTELLGRAPH_SRC_SOLVER_MOMENTUM_H_
#define INTELLGRAPH_SRC_SOLVER_MOMENTUM_H_

#include "src/edge.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/solver.h"

namespace intellgraph {

// Class that implements the Stochastic Gradient Descent algorithm with momentum
template <typename T> class Momentum : public Solver<T> {
public:
  explicit Momentum(T eta, T gamma = 0.9, T lambda = 0);
  ~Momentum() override;

  void Visit(Edge<T> &edge) override;

private:
  T eta_ = 0;
  T gamma_ = 0;
  T lambda_ = 0;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class Momentum<float>;
extern template class Momentum<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_SOLVER_MOMENTUM_H_

