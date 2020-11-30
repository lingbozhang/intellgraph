/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.  Licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
        Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#include "src/edge/vertex/output_vertex_impl.cc"

#include "src/edge/vertex/sigmoid_l2.h"
#include "src/eigen.h"
#include "gtest/gtest.h"

namespace intellgraph {
namespace {

TEST(OpVertexImplTest, ResizeVertexSuccess) {
  OutputVertexImpl<float, SigmoidL2> output_vertex_float(0, 1, 1);
  OutputVertexImpl<double, SigmoidL2> output_vertex_double(0, 1, 1);

  output_vertex_float.ResizeVertex(2);
  output_vertex_double.ResizeVertex(2);

  EXPECT_FLOAT_EQ(output_vertex_float.col(), 2);
  EXPECT_FLOAT_EQ(output_vertex_float.activation().cols(), 2);
  EXPECT_DOUBLE_EQ(output_vertex_double.col(), 2);
  EXPECT_DOUBLE_EQ(output_vertex_double.activation().cols(), 2);
}

TEST(OutputVertexImplTest, GetIdSuccess) {
  OutputVertexImpl<float, SigmoidL2> output_vertex_float(1, 1, 1);
  OutputVertexImpl<double, SigmoidL2> output_vertex_double(1, 1, 1);

  EXPECT_EQ(output_vertex_float.id(), 1);
  EXPECT_EQ(output_vertex_double.id(), 1);
}

} // namespace
} // namespace intellgraph
