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
#include "src/edge/vertex/input_vertex_impl.cc"

#include "glog/logging.h"
#include "src/eigen.h"
#include "gtest/gtest.h"

namespace intellgraph {
namespace {

TEST(InputVertexImplTest, GetIdSuccess) {
  InputVertexImpl<float, DummyTransformer> input_vertex_float(1, 1, 1);
  InputVertexImpl<double, DummyTransformer> input_vertex_double(1, 1, 1);

  EXPECT_EQ(input_vertex_float.id(), 1);
  EXPECT_EQ(input_vertex_double.id(), 1);
}

TEST(InputVertexImplTest, GetRowSuccess) {
  InputVertexImpl<float, DummyTransformer> input_vertex_float(0, 10, 1);
  InputVertexImpl<double, DummyTransformer> input_vertex_double(0, 10, 1);

  EXPECT_EQ(input_vertex_float.row(), 10);
  EXPECT_EQ(input_vertex_double.row(), 10);
}

TEST(InputVertexImplTest, GetColSuccess) {
  InputVertexImpl<float, DummyTransformer> input_vertex_float(0, 1, 10);
  InputVertexImpl<double, DummyTransformer> input_vertex_double(0, 1, 10);

  EXPECT_EQ(input_vertex_float.col(), 10);
  EXPECT_EQ(input_vertex_double.col(), 10);
}

TEST(InputVertexImplTest, SetFeatureSuccess) {
  InputVertexImpl<float, DummyTransformer> input_vertex_float(0, 1, 1);
  InputVertexImpl<double, DummyTransformer> input_vertex_double(0, 1, 1);

  MatrixX<float> data_float = MatrixX<float>::Constant(1, 1, 1);
  MatrixX<double> data_double = MatrixX<double>::Constant(1, 1, 1);

  input_vertex_float.set_feature(&data_float);
  input_vertex_double.set_feature(&data_double);

  EXPECT_EQ(input_vertex_float.act()(0, 0), 1.0f);
  EXPECT_EQ(input_vertex_double.act()(0, 0), 1.0);
}

} // namespace
} // namespace intellgraph
