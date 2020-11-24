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
#include "src/visitor/backward_visitor.h"

#include "src/edge/vertex/op_vertex_impl.h"
#include "src/edge/vertex/sigmoid.h"
#include "gtest/gtest.h"

namespace intellgraph {
namespace {

TEST(BackwardVisitorTest, VisitSuccess) {
  OpVertexImpl<float, Sigmoid> vtx_in_float(0, 2, 2);
  OpVertexImpl<double, Sigmoid> vtx_in_double(0, 2, 2);
  OpVertexImpl<float, Sigmoid> vtx_out_float(1, 4, 2);
  OpVertexImpl<double, Sigmoid> vtx_out_double(1, 4, 2);

  DenseEdgeImpl<float, OpVertex<float>> edge_float(0, &vtx_in_float,
                                                   &vtx_out_float);
  DenseEdgeImpl<double, OpVertex<double>> edge_double(0, &vtx_in_double,
                                                      &vtx_out_double);

  vtx_in_float.mutable_activation()->setConstant(1.5f);
  vtx_in_double.mutable_activation()->setConstant(1.5);
  MatrixX<float> act_in_float = *vtx_in_float.mutable_activation();
  MatrixX<double> act_in_double = *vtx_in_double.mutable_activation();

  vtx_out_float.mutable_delta()->setConstant(1.0f);
  vtx_out_double.mutable_delta()->setConstant(1.0);
  MatrixX<float> delta_out_float = *vtx_out_float.mutable_delta();
  MatrixX<double> delta_out_double = *vtx_out_double.mutable_delta();

  edge_float.mutable_weight()->setConstant(2.0f);
  edge_double.mutable_weight()->setConstant(2.0);
  MatrixX<float> weight_float = *edge_float.mutable_weight();
  MatrixX<double> weight_double = *edge_double.mutable_weight();

  BackwardVisitor<float> visitor_float;
  BackwardVisitor<double> visitor_double;
  edge_float.Accept(visitor_float);
  edge_double.Accept(visitor_double);

  EXPECT_EQ(*edge_float.mutable_nabla_weight(),
            act_in_float.matrix() * delta_out_float.transpose() / 2.0f);
  EXPECT_EQ(*edge_double.mutable_nabla_weight(),
            act_in_double.matrix() * delta_out_double.transpose() / 2.0);

  EXPECT_EQ(*vtx_in_float.mutable_delta(),
            ((weight_float * delta_out_float).array() *
             (act_in_float.array() * (1.0f - act_in_float.array())))
                .matrix());
  EXPECT_EQ(*vtx_in_double.mutable_delta(),
            ((weight_double * delta_out_double).array() *
             (act_in_double.array() * (1.0 - act_in_double.array())))
                .matrix());
}

} // namespace
} // namespace intellgraph
