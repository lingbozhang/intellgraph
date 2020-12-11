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
#include "src/edge/vertex/relu.h"

#include <cmath>

#include "src/edge/vertex/op_vertex_impl.h"
#include "gtest/gtest.h"

namespace intellgraph {
namespace {

TEST(ReluTest, ActivateSuccess) {
  // Activation element value EQ zero
  OpVertexImpl<float, Relu> op_vertex_float(0, 1, 1);
  op_vertex_float.Activate();
  EXPECT_FLOAT_EQ(op_vertex_float.act()(0, 0), 0.0f);

  OpVertexImpl<double, Relu> op_vertex_double(1, 1, 1);
  op_vertex_double.Activate();
  EXPECT_DOUBLE_EQ(op_vertex_double.act()(0, 0), 0.0);

  // Activation element value GT zero
  op_vertex_float.mutable_act().setIdentity();
  op_vertex_float.Activate();
  EXPECT_FLOAT_EQ(op_vertex_float.act()(0, 0), 1.0f);

  op_vertex_double.mutable_act().setIdentity();
  op_vertex_double.Activate();
  EXPECT_DOUBLE_EQ(op_vertex_double.act()(0, 0), 1.0);

  // Activation element value LT zero
  op_vertex_float.mutable_act().setConstant(-1.0f);
  op_vertex_float.Activate();
  EXPECT_FLOAT_EQ(op_vertex_float.act()(0, 0), 0.0f);

  op_vertex_double.mutable_act().setConstant(-1.0);
  op_vertex_double.Activate();
  EXPECT_DOUBLE_EQ(op_vertex_double.act()(0, 0), 0.0);
}

TEST(ReluTest, DeriveSuccess) {
  // Activation element value EQ zero
  OpVertexImpl<float, Relu> op_vertex_float(0, 1, 1);
  op_vertex_float.mutable_act().setConstant(0.0f);
  op_vertex_float.Derive();
  EXPECT_FLOAT_EQ(op_vertex_float.act()(0, 0), 0.0f);

  OpVertexImpl<double, Relu> op_vertex_double(1, 1, 1);
  op_vertex_float.mutable_act().setConstant(0.0);
  op_vertex_double.Derive();
  EXPECT_DOUBLE_EQ(op_vertex_double.act()(0, 0), 0.0);

  // Activation element value GT zero
  op_vertex_float.mutable_act().setConstant(2.0f);
  op_vertex_float.Derive();
  EXPECT_FLOAT_EQ(op_vertex_float.act()(0, 0), 1.0f);

  op_vertex_double.mutable_act().setConstant(2.0);
  op_vertex_double.Derive();
  EXPECT_DOUBLE_EQ(op_vertex_double.act()(0, 0), 1.0);
}

} // namespace
} // namespace intellgraph
