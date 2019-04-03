/* Copyright 2019 The Nicole Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "edge/dense_edge.h"
#include "edge/edge_factory.h"
#include "edge/edge_parameter.h"
#include "edge/edge.h"
#include "node/act_loss_node.h"
#include "node/activation_node.h"
#include "node/node.h"
#include "node/node_factory.h"
#include "node/node_parameter.h"
#include "node/output_node.h"
#include "node/sigmoid_node.h"
#include "node/sigmoid_l2_node.h"
#include "utility/registry.h"
#include "utility/random.h"

using namespace std;
using namespace intellgraph;

// EdgeTest contains tests for programs in edge directory
class EdgeTest : public ::testing::Test {
 protected:
  EdgeTest() {
    Registry::LoadEdgeRegistry();
    trainning1_data_ptr_ = make_shared<MatXX<float>>(2, 1);
    trainning1_label_ptr_ = make_shared<MatXX<float>>(1, 1);

    trainning2_data_ptr_ = make_shared<MatXX<float>>(2, 2);
    trainning2_label_ptr_ = make_shared<MatXX<float>>(1, 2);

    weight1_ptr_v1_ = make_unique<MatXX<float>>(2, 3);
    weight2_ptr_v1_ = make_unique<MatXX<float>>(3, 1);

    bias1_ptr_v1_ = make_unique<MatXX<float>>(3, 1);
    bias2_ptr_v1_ = make_unique<MatXX<float>>(1, 1);
  
    weight1_ptr_v2_ = make_unique<MatXX<float>>(2, 3);
    weight2_ptr_v2_ = make_unique<MatXX<float>>(3, 1);
  
    bias1_ptr_v2_ = make_unique<MatXX<float>>(3, 2);
    bias2_ptr_v2_ = make_unique<MatXX<float>>(1, 2);

    // Data
    trainning2_data_ptr_->array() << 1.0, 3.0,
                                    2.0, 5.0;
    trainning2_label_ptr_->array() << 1.0, 2.0;

    trainning1_data_ptr_->array() << 1.0,
                                     2.0;
    trainning1_label_ptr_->array() << 1.0;

    weight1_ptr_v1_->array() << 1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0;

    weight2_ptr_v1_->array() << 1.0,
                                2.0,
                                3.0;
    
    bias1_ptr_v1_->array() << 1.0,
                              2.0,
                              3.0;

    bias2_ptr_v1_->array() << 1.0;

    weight1_ptr_v2_->array() << 1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0;

    weight2_ptr_v2_->array() << 1.0,
                                2.0,
                                3.0;
    
    bias1_ptr_v2_->array() << 1.0, 1.0,
                              2.0, 2.0,
                              3.0, 3.0;

    bias2_ptr_v2_->array() << 1.0, 1.0;  
    // Support objects
    input_node_ptr_v1_ = NodeFactory<float, InputNode<float>>::Instantiate( \
        node_param1_);
    inner_node_ptr_v1_ = NodeFactory<float, Node<float>>::Instantiate( \
        node_param3_);
    output_node_ptr_v1_ = NodeFactory<float, OutputNode<float>>::Instantiate( \
        node_param5_);

    input_node_ptr_v2_ = NodeFactory<float, InputNode<float>>::Instantiate( \
        node_param2_);
    inner_node_ptr_v2_ = NodeFactory<float, Node<float>>::Instantiate( \
        node_param4_);
    output_node_ptr_v2_ = NodeFactory<float, OutputNode<float>>::Instantiate( \
        node_param6_);
    // Test objects
    edge1_ptr_v1_ = EdgeFactory<float, Edge<float>>::Instantiate(edge1_param_v1_);
    edge2_ptr_v1_ = EdgeFactory<float, Edge<float>>::Instantiate(edge2_param_v1_);
    edge1_ptr_v2_ = EdgeFactory<float, Edge<float>>::Instantiate(edge1_param_v2_);
    edge2_ptr_v2_ = EdgeFactory<float, Edge<float>>::Instantiate(edge2_param_v2_);
  }
  ~EdgeTest() {}

  NodeParameter<float> node_param1_{0, "SigInputNode", {2, 1}};
  NodeParameter<float> node_param2_{1, "SigInputNode", {2, 2}};

  NodeParameter<float> node_param3_{2, "SigmoidNode", {3, 1}};
  NodeParameter<float> node_param4_{3, "SigmoidNode", {3, 2}};

  NodeParameter<float> node_param5_{4, "SigL2Node", {1, 1}};
  NodeParameter<float> node_param6_{5, "SigL2Node", {1, 2}};

  // Data
  MatXXSPtr<float> trainning1_data_ptr_{nullptr};
  MatXXSPtr<float> trainning1_label_ptr_{nullptr};

  MatXXSPtr<float> trainning2_data_ptr_{nullptr};
  MatXXSPtr<float> trainning2_label_ptr_{nullptr};

  MatXXSPtr<float> weight1_ptr_v1_{nullptr};
  MatXXSPtr<float> weight2_ptr_v1_{nullptr};

  MatXXSPtr<float> bias1_ptr_v1_{nullptr};
  MatXXSPtr<float> bias2_ptr_v1_{nullptr};

  MatXXSPtr<float> weight1_ptr_v2_{nullptr};
  MatXXSPtr<float> weight2_ptr_v2_{nullptr};

  MatXXSPtr<float> bias1_ptr_v2_{nullptr};
  MatXXSPtr<float> bias2_ptr_v2_{nullptr};
  // Support objects
  InputNodeUPtr<float> input_node_ptr_v1_{nullptr};
  NodeUPtr<float> inner_node_ptr_v1_{nullptr};
  OutputNodeUPtr<float> output_node_ptr_v1_{nullptr};

  InputNodeUPtr<float> input_node_ptr_v2_{nullptr};
  NodeUPtr<float> inner_node_ptr_v2_{nullptr};
  OutputNodeUPtr<float> output_node_ptr_v2_{nullptr};

  // Test objects
  EdgeParameter edge1_param_v1_{0, "DenseEdge", {2, 1}, {3, 1}};
  EdgeParameter edge2_param_v1_{1, "DenseEdge", {3, 1}, {1, 1}};

  EdgeParameter edge1_param_v2_{2, "DenseEdge", {2, 2}, {3, 2}};
  EdgeParameter edge2_param_v2_{3, "DenseEdge", {3, 2}, {1, 2}};

  EdgeUPtr<float> edge1_ptr_v1_{nullptr};
  EdgeUPtr<float> edge2_ptr_v1_{nullptr};
  EdgeUPtr<float> edge1_ptr_v2_{nullptr};
  EdgeUPtr<float> edge2_ptr_v2_{nullptr};

  const double kAbsoluteError_ = 1.0E-7;
  const double kRelativeError_ = 1.0E-6;
};

TEST_F(EdgeTest, TestForward_mute) {
  // version 1
  input_node_ptr_v1_->FeedFeature_k(trainning1_data_ptr_);

  edge1_ptr_v1_->get_c_weight_ptr()->array() = weight1_ptr_v1_->array();
  edge2_ptr_v1_->get_c_weight_ptr()->array() = weight2_ptr_v1_->array();
  inner_node_ptr_v1_->get_c_bias_ptr()->array() = bias1_ptr_v1_->array();
  output_node_ptr_v1_->get_c_bias_ptr()->array() = bias2_ptr_v1_->array();

  edge1_ptr_v1_->Forward_mute(input_node_ptr_v1_.get(), inner_node_ptr_v1_.get());
  edge2_ptr_v1_->Forward_mute(inner_node_ptr_v1_.get(), output_node_ptr_v1_.get());

  MatXX<float> stage1 = weight1_ptr_v1_->transpose() * \
                        trainning1_data_ptr_->matrix() + bias1_ptr_v1_->matrix();
  MatXX<float> stage2 = weight2_ptr_v1_->transpose() * \
                        stage1 + bias2_ptr_v1_->matrix();
  float correct_value = stage2(0, 0);
  float test_value = output_node_ptr_v1_->get_c_activation_ptr()->array()(0, 0);
  EXPECT_NEAR(test_value, correct_value, kAbsoluteError_);

  // version 2
  input_node_ptr_v2_->FeedFeature_k(trainning2_data_ptr_);

  edge1_ptr_v2_->get_c_weight_ptr()->array() = weight1_ptr_v2_->array();
  edge2_ptr_v2_->get_c_weight_ptr()->array() = weight2_ptr_v2_->array();
  inner_node_ptr_v2_->get_c_bias_ptr()->array() = bias1_ptr_v2_->array();
  output_node_ptr_v2_->get_c_bias_ptr()->array() = bias2_ptr_v2_->array();

  edge1_ptr_v2_->Forward_mute(input_node_ptr_v2_.get(), inner_node_ptr_v2_.get());
  edge2_ptr_v2_->Forward_mute(inner_node_ptr_v2_.get(), output_node_ptr_v2_.get());

  stage1 = weight1_ptr_v2_->transpose() * \
           trainning2_data_ptr_->matrix() + bias1_ptr_v2_->matrix();
  stage2 = weight2_ptr_v2_->transpose() * \
           stage1 + bias2_ptr_v2_->matrix();
  correct_value = stage2(0, 0);
  test_value = output_node_ptr_v2_->get_c_activation_ptr()->array()(0, 0);
  EXPECT_NEAR(test_value, correct_value, kAbsoluteError_);
  correct_value = stage2(0, 1);
  test_value = output_node_ptr_v2_->get_c_activation_ptr()->array()(0, 1);
  EXPECT_NEAR(test_value, correct_value, kAbsoluteError_);
}

TEST_F(EdgeTest, TestBackward_mute) {
  // version 1
  input_node_ptr_v1_->FeedFeature_k(trainning1_data_ptr_);

  output_node_ptr_v1_->CallActFxn();
  output_node_ptr_v1_->CalcDelta_k(trainning1_label_ptr_.get());
  float test_value = output_node_ptr_v1_->get_c_delta_ptr()->array()(0, 0);
  float correct_value = -0.25;
  EXPECT_NEAR(test_value, correct_value, kAbsoluteError_);

  edge1_ptr_v1_->get_c_weight_ptr()->array() = weight1_ptr_v1_->array();
  edge2_ptr_v1_->get_c_weight_ptr()->array() = weight2_ptr_v1_->array();
  inner_node_ptr_v1_->get_c_bias_ptr()->array() = bias1_ptr_v1_->array();
  inner_node_ptr_v1_->CallActFxn();

  output_node_ptr_v1_->get_c_bias_ptr()->array() = bias2_ptr_v1_->array();

  edge2_ptr_v1_->Backward_mute(inner_node_ptr_v1_.get(), output_node_ptr_v1_.get());
  for (int i = 0; i < 3; ++i) {
    test_value = edge2_ptr_v1_->get_c_nabla_weight_ptr()->array()(i, 0);
    correct_value = -0.5f * 0.25f;
    EXPECT_NEAR(test_value, correct_value, kAbsoluteError_);
    test_value = inner_node_ptr_v1_->get_c_delta_ptr()->array()(i, 0);
    correct_value = (i + 1) * -0.25f * 0.25f;
    EXPECT_NEAR(test_value, correct_value, kAbsoluteError_);
  }
}