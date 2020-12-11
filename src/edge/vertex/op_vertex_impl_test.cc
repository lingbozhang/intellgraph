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
#include "src/edge/vertex/op_vertex_impl.cc"

#include "glog/logging.h"
#include "src/edge/vertex/op_vertex_impl.h"
#include "src/edge/vertex/sigmoid.h"
#include "src/eigen.h"
#include "gtest/gtest.h"

namespace intellgraph {
namespace {

TEST(OpVertexImplTest, ResizeVertexSuccess) {
  OpVertexImpl<float, Sigmoid> op_vertex_float(0, 1, 1);
  OpVertexImpl<double, Sigmoid> op_vertex_double(0, 1, 1);

  op_vertex_float.ResizeVertex(2);
  op_vertex_double.ResizeVertex(2);

  EXPECT_FLOAT_EQ(op_vertex_float.col(), 2);
  EXPECT_FLOAT_EQ(op_vertex_float.act().cols(), 2);
  EXPECT_DOUBLE_EQ(op_vertex_double.col(), 2);
  EXPECT_DOUBLE_EQ(op_vertex_double.act().cols(), 2);
}

TEST(OpVertexImplTest, GetIdSuccess) {
  OpVertexImpl<float, Sigmoid> op_vertex_float(1, 1, 1);
  OpVertexImpl<double, Sigmoid> op_vertex_double(1, 1, 1);

  EXPECT_EQ(op_vertex_float.id(), 1);
  EXPECT_EQ(op_vertex_double.id(), 1);
}

TEST(OpVertexImplTest, GetRowSuccess) {
  OpVertexImpl<float, Sigmoid> op_vertex_float(1, 10, 1);
  OpVertexImpl<double, Sigmoid> op_vertex_double(1, 10, 1);

  EXPECT_EQ(op_vertex_float.row(), 10);
  EXPECT_EQ(op_vertex_double.row(), 10);
}

TEST(OpVertexImplTest, GetColSuccess) {
  OpVertexImpl<float, Sigmoid> op_vertex_float(1, 1, 10);
  OpVertexImpl<double, Sigmoid> op_vertex_double(1, 1, 10);

  EXPECT_EQ(op_vertex_float.col(), 10);
  EXPECT_EQ(op_vertex_double.col(), 10);
}

} // namespace
} // namespace intellgraph
