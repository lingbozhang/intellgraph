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
#include "src/visitor/forward_visitor.h"

#include "src/edge/dense_edge_impl.h"
#include "src/edge/vertex/op_vertex_impl.h"
#include "src/edge/vertex/sigmoid.h"
#include "src/eigen.h"
#include "gtest/gtest.h"

namespace intellgraph {
namespace {

TEST(ForwardVisitorTest, VisitSuccess) {
  OpVertexImpl<float, Sigmoid> vtx_in_float(0, 2, 2);
  OpVertexImpl<double, Sigmoid> vtx_in_double(0, 2, 2);
  OpVertexImpl<float, Sigmoid> vtx_out_float(1, 4, 2);
  OpVertexImpl<double, Sigmoid> vtx_out_double(1, 4, 2);

  DenseEdgeImpl<float, OpVertex<float>> edge_float(0, &vtx_in_float,
                                                   &vtx_out_float);
  DenseEdgeImpl<double, OpVertex<double>> edge_double(0, &vtx_in_double,
                                                      &vtx_out_double);

  vtx_out_float.mutable_bias()->setConstant(1.0f);
  vtx_out_double.mutable_bias()->setConstant(1.0f);

  ForwardVisitor<float> visitor_float;
  ForwardVisitor<double> visitor_double;
  edge_float.Accept(visitor_float);
  edge_double.Accept(visitor_double);

  Eigen::Matrix<float, 4, 2> expected_result_float;
  expected_result_float << 1.5f, 1.5f, 1.5f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f;
  EXPECT_EQ((*vtx_out_float.mutable_activation()), expected_result_float);

  Eigen::Matrix<double, 4, 2> expected_result_double;
  expected_result_double << 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0;
  EXPECT_EQ((*vtx_out_double.mutable_activation()), expected_result_double);
}

} // namespace
} // namespace intellgraph
