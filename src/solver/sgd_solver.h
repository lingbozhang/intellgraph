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
#ifndef INTELLGRAPH_SRC_SOLVER_SGD_SOLVER_H_
#define INTELLGRAPH_SRC_SOLVER_SGD_SOLVER_H_

#include "src/solver.h"

namespace intellgraph {

// Class that implements the Stochastic Gradient Descent algorithm
template <typename T> class SgdSolver : public Solver<T> {
public:
  explicit SgdSolver(T eta, T lambda);
  ~SgdSolver() override;

  void Visit(DenseEdgeImpl<T, OpVertex<T>, OpVertex<T>> &edge) override;

private:
  T eta_ = 0;
  T lambda_ = 0;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class SgdSolver<float>;
extern template class SgdSolver<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_SOLVER_SGD_SOLVER_H_
