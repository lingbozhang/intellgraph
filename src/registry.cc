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
#include "src/registry.h"

#include "glog/logging.h"
#include "src/edge.h"
#include "src/edge/dense_edge_impl.h"
#include "src/edge/op_vertex.h"
#include "src/edge/output_vertex.h"
#include "src/edge/seq_output.h"
#include "src/edge/seq_vertex.h"
#include "src/edge/vertex/cross_entropy.h"
#include "src/edge/vertex/input_vertex.h"
#include "src/edge/vertex/input_vertex_impl.h"
#include "src/edge/vertex/op_vertex_impl.h"
#include "src/edge/vertex/output_vertex_impl.h"
#include "src/edge/vertex/relu.h"
#include "src/edge/vertex/seq_output_impl.h"
#include "src/edge/vertex/seq_vertex_impl.h"
#include "src/edge/vertex/sigmoid.h"
#include "src/edge/vertex/sigmoid_l2.h"
#include "src/factory.h"
#include "src/solver.h"
#include "src/solver/sgd_solver.h"

namespace intellgraph {

// static
void Registry::LoadRegistry() {
  LOG(INFO) << "Registering the Relu vertex...";
  REGISTER_VERTEX(OpVertex, OpVertexImpl, Relu);
  REGISTER_VERTEX(SeqVertex, SeqVertexImpl, Relu);

  LOG(INFO) << "Registering the Sigmoid vertex...";
  REGISTER_VERTEX(OpVertex, OpVertexImpl, Sigmoid);
  REGISTER_VERTEX(SeqVertex, SeqVertexImpl, Sigmoid);

  LOG(INFO) << "Registering the SigmoidL2 ouput vertex...";
  REGISTER_VERTEX(OutputVertex, OutputVertexImpl, SigmoidL2);
  REGISTER_VERTEX(SeqOutput, SeqOutputImpl, SigmoidL2);

  LOG(INFO) << "Registering the CrossEntropy ouput vertex...";
  REGISTER_VERTEX(OutputVertex, OutputVertexImpl, CrossEntropy);
  REGISTER_VERTEX(SeqOutput, SeqOutputImpl, CrossEntropy);

  LOG(INFO) << "Registering the Input vertex...";
  REGISTER_VERTEX(InputVertex, InputVertexImpl, DummyTransformer);

  LOG(INFO) << "Registering the Dense edge...";
  REGISTER_EDGE(Edge, DenseEdgeImpl, OpVertex, OpVertex, Dense);

  LOG(INFO) << "Registering the Stochastic Gradient Descent solver...";
  REGISTER_SOLVER(Solver, SgdSolver, SGD);
}

} // namespace intellgraph
