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
#include "src/edge/vertex/sigmoid_l2.h"

#include <cmath>

#include "src/edge/vertex/output_vertex_impl.h"
#include "src/edge/vertex/sigmoid_l2.h"
#include "gtest/gtest.h"

namespace intellgraph {
namespace {

TEST(SigmoidL2Test, ActivateSuccess) {
  // Activation element value EQ zero
  OutputVertexImpl<float, SigmoidL2> output_vertex_float(0, 1, 1);
  OutputVertexImpl<double, SigmoidL2> output_vertex_double(0, 1, 1);

  output_vertex_float.Activate();
  output_vertex_double.Activate();

  EXPECT_FLOAT_EQ((*output_vertex_float.mutable_activation())(0, 0), 0.5f);
  EXPECT_DOUBLE_EQ((*output_vertex_double.mutable_activation())(0, 0), 0.5);

  // Activation element value GT zero
  output_vertex_float.mutable_activation()->setIdentity();
  output_vertex_double.mutable_activation()->setIdentity();

  output_vertex_float.Activate();
  output_vertex_double.Activate();

  EXPECT_FLOAT_EQ((*output_vertex_float.mutable_activation())(0, 0),
                  1.0f / (1.0f + std::exp(-1.0f)));
  EXPECT_DOUBLE_EQ((*output_vertex_double.mutable_activation())(0, 0),
                   1.0 / (1.0 + std::exp(-1.0)));

  // Activation element value LT zero
  output_vertex_float.mutable_activation()->setConstant(-1.0f);
  output_vertex_double.mutable_activation()->setConstant(-1.0);

  output_vertex_float.Activate();
  output_vertex_double.Activate();

  EXPECT_FLOAT_EQ((*output_vertex_float.mutable_activation())(0, 0),
                  1.0f / (1.0f + std::exp(1.0f)));
  EXPECT_DOUBLE_EQ((*output_vertex_double.mutable_activation())(0, 0),
                   1.0 / (1.0 + std::exp(1.0)));
}

TEST(SigmoidL2Test, DeriveSuccess) {
  OutputVertexImpl<float, SigmoidL2> output_vertex_float(0, 1, 1);
  OutputVertexImpl<double, SigmoidL2> output_vertex_double(0, 1, 1);

  output_vertex_float.Activate();
  output_vertex_double.Activate();

  output_vertex_float.Derive();
  output_vertex_double.Derive();

  EXPECT_FLOAT_EQ((*output_vertex_float.mutable_activation())(0, 0), 0.25f);
  EXPECT_DOUBLE_EQ((*output_vertex_double.mutable_activation())(0, 0), 0.25);
}

TEST(SigmoidL2Test, CalcLossSuccess) {
  OutputVertexImpl<float, SigmoidL2> output_vertex_float(0, 4, 2);
  OutputVertexImpl<double, SigmoidL2> output_vertex_double(0, 4, 2);

  MatrixX<float> labels_float = MatrixX<float>::Constant(4, 2, 1);
  MatrixX<double> labels_double = MatrixX<double>::Constant(4, 2, 1);

  EXPECT_FLOAT_EQ(output_vertex_float.CalcLoss(labels_float), 2.0f);
  EXPECT_DOUBLE_EQ(output_vertex_double.CalcLoss(labels_double), 2.0);
}

TEST(SigmoidL2Test, CalcDeltaSuccess) {
  OutputVertexImpl<float, SigmoidL2> output_vertex_float(0, 1, 1);
  OutputVertexImpl<double, SigmoidL2> output_vertex_double(0, 1, 1);

  MatrixX<float> *activation_float = output_vertex_float.mutable_activation();
  MatrixX<double> *activation_double =
      output_vertex_double.mutable_activation();

  activation_float->setConstant(0.5f);
  activation_double->setConstant(0.5);

  MatrixX<float> labels_float = MatrixX<float>::Constant(1, 1, 2.0f);
  MatrixX<double> labels_double = MatrixX<double>::Constant(1, 1, 2.0);

  output_vertex_float.CalcDelta(labels_float);
  output_vertex_double.CalcDelta(labels_double);

  EXPECT_FLOAT_EQ((*output_vertex_float.mutable_delta())(0, 0), -1.5f * 0.25f);
  EXPECT_DOUBLE_EQ((*output_vertex_double.mutable_delta())(0, 0), -1.5 * 0.25);
}

} // namespace
} // namespace intellgraph
