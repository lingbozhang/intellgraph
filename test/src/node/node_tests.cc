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

#include <cmath>
#include <iostream>

#include "gtest/gtest.h"
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

// NodeTest contains tests for programs in node directory
class NodeTest : public ::testing::Test {
 protected:
  NodeTest() {
    Registry::LoadNodeRegistry();
    auto node_param1 = NodeParameter<float>(0, "SigmoidNode", {100,1});
    auto node_param2 = NodeParameter<float>(1, "SigL2Node", {100,1});

    sigmoid_node_ptr_ = std::move( \
        NodeFactory<float, Node<float>>::Instantiate(node_param1));
    sigmoid_l2_node_ptr_ = std::move( \
        NodeFactory<float, OutputNode<float>>::Instantiate(node_param2));
    
    node_ptr_v_.push_back(sigmoid_node_ptr_.get());
    node_ptr_v_.push_back(sigmoid_l2_node_ptr_.get());
  }
  ~NodeTest() {}
  NodeUPtr<float> sigmoid_node_ptr_;
  OutputNodeUPtr<float> sigmoid_l2_node_ptr_;
  std::vector<Node<float>*> node_ptr_v_;

  const double kAbsoluteError_ = 1.0E-7;
  const double kRelativeError_ = 1.0E-6;
};

TEST_F(NodeTest, TestCallActFxn) {
  std::vector<float> evaluated_values = {-1.0E6, -100, -100.0, -1.0E-7, 0.0, \
                                         1.0E-6, 100, 100.0, 1.0E-7};
  for (auto& node_ptr : node_ptr_v_) {
    for (auto value : evaluated_values) {
      //node_ptr->PrintAct();
      node_ptr->set_c_activation(value);
      node_ptr->CallActFxn();
      float correct_value;
      if (value >= 0) {
        correct_value = 1.0 / (1.0 + std::exp(-value));
      } else {
        correct_value = std::exp(value) / (1.0 + std::exp(value));
      }
      for (size_t i = 0; i < node_ptr->get_k_dims()[0]; ++i) {
        float test_value = node_ptr->get_c_activation_ptr()->array()(i);
        EXPECT_NEAR(test_value, correct_value, kAbsoluteError_)
            << "Evaluated value: " << value;
      }
    }
  }
  // Tests with random initialized data
  for (auto& node_ptr : node_ptr_v_) {
    node_ptr->ApplyUnaryFunctor_k(NormalFunctor<float>(0.0, 1.0));
    MatXXSPtr<float> correct_data_ptr = make_shared<MatXX<float>>(100, 1);
    correct_data_ptr->array() = node_ptr->get_c_activation_ptr()->array();
    
    node_ptr->CallActFxn();

    for (size_t i = 0; i < node_ptr->get_k_dims()[0]; ++i) {
      float correct_value;
      float test_value = node_ptr->get_c_activation_ptr()->array()(i);
      float value = correct_data_ptr->array()(i);
      if (value >= 0) {
        correct_value = 1.0 / (1.0 + std::exp(-value));
      } else {
        correct_value = std::exp(value) / (1.0 + std::exp(value));
      }
      EXPECT_NEAR(test_value, correct_value, kAbsoluteError_)
                  << "Evaluated value: " << value; 
    }
  }
}


TEST_F(NodeTest, TestCalcPrime) {
  std::vector<float> evaluated_values = {-1.0E6, -100, -100.0, -1.0E-7, 0.0, \
                                         1.0E-6, 100, 100.0, 1.0E-7};
  for (auto& node_ptr : node_ptr_v_) {
    for (auto value : evaluated_values) {
      //node_ptr->PrintAct();
      node_ptr->set_c_activation(value);
      node_ptr->CalcActPrime();
      float correct_value;
      if (value >= 0) {
        correct_value = 1.0 / (1.0 + std::exp(-value));
      } else {
        correct_value = std::exp(value) / (1.0 + std::exp(value));
      }
      correct_value = correct_value * (1.0 - correct_value);
      for (size_t i = 0; i < node_ptr->get_k_dims()[0]; ++i) {
        float test_value = node_ptr->get_c_activation_ptr()->array()(i);
        EXPECT_NEAR(test_value, correct_value, kAbsoluteError_)
            << "Evaluated value: " << value;
      }
    }
  }
  // Tests with random initialized data
  for (auto& node_ptr : node_ptr_v_) {
    node_ptr->ApplyUnaryFunctor_k(NormalFunctor<float>(0.0, 1.0));
    MatXXSPtr<float> correct_data_ptr = make_shared<MatXX<float>>(100, 1);
    correct_data_ptr->array() = node_ptr->get_c_activation_ptr()->array();
    
    node_ptr->CalcActPrime();

    for (size_t i = 0; i < node_ptr->get_k_dims()[0]; ++i) {
      float correct_value;
      float test_value = node_ptr->get_c_activation_ptr()->array()(i);
      float value = correct_data_ptr->array()(i);
      if (value >= 0) {
        correct_value = 1.0 / (1.0 + std::exp(-value));
      } else {
        correct_value = std::exp(value) / (1.0 + std::exp(value));
      }
      correct_value = correct_value * (1.0 - correct_value);
      EXPECT_NEAR(test_value, correct_value, kAbsoluteError_)
                  << "Evaluated value: " << value; 
    }
  }
}

TEST_F(NodeTest, TestCalcLoss) {
  std::vector<float> evaluated_values = {-1.0E6, -100, -100.0, -1.0E-7, 0.0, \
                                         1.0E-6, 100, 100.0, 1.0E-7};
  for (auto value : evaluated_values) {
    sigmoid_l2_node_ptr_->set_c_activation(value);
    float activation;
    if (value >= 0) {
      activation = 1.0 / (1.0 + std::exp(-value));
    } else {
      activation = std::exp(value) / (1.0 + std::exp(value));
    }
    MatXXUPtr<float> data_result_ptr = make_unique<MatXX<float>>(100, 1);
    data_result_ptr->array() = data_result_ptr->array().unaryExpr( \
        std::function<float(float)>(NormalFunctor<float>(0.0, 1.0)));

    double correct_loss = 0.0;
    for (size_t i = 0; i < data_result_ptr->array().rows(); ++i) {
      correct_loss += (activation - data_result_ptr->array()(i)) * \
                      (activation - data_result_ptr->array()(i));
    }
    float test_loss = sigmoid_l2_node_ptr_->CalcLoss_k(*data_result_ptr);
    EXPECT_NEAR(test_loss, correct_loss, kRelativeError_ * abs(correct_loss))
        << "Activation value: " << activation << std::endl;
  }
}

TEST_F(NodeTest, TestCalcDelta) {
  std::vector<float> evaluated_values = {-1.0E6, -100, -100.0, -1.0E-7, 0.0, \
                                         1.0E-6, 100, 100.0, 1.0E-7};
  for (auto value : evaluated_values) {
    sigmoid_l2_node_ptr_->set_c_activation(value);
    MatXXUPtr<float> correct_act_ptr = make_unique<MatXX<float>>(100,1);
    MatXXUPtr<float> correct_prime_ptr = make_unique<MatXX<float>>(100,1);
    MatXXUPtr<float> correct_delta_ptr = make_unique<MatXX<float>>(100,1);
    float activation;
    if (value >= 0) {
      activation = 1.0 / (1.0 + std::exp(-value));
      correct_act_ptr->array() = activation;
      correct_prime_ptr->array() = activation * (1.0 - activation);
    } else {
      activation = std::exp(value) / (1.0 + std::exp(value));
      correct_act_ptr->array() = activation;
      correct_prime_ptr->array() = activation * (1.0 - activation);
    }
    MatXXUPtr<float> data_result_ptr = make_unique<MatXX<float>>(100, 1);
    data_result_ptr->array() = data_result_ptr->array().unaryExpr( \
        std::function<float(float)>(NormalFunctor<float>(0.0, 1.0)));

    correct_delta_ptr->array() = 2.0 * (correct_act_ptr->array() - \
                                 data_result_ptr->array()) * \
                                 correct_prime_ptr->array();

    sigmoid_l2_node_ptr_->CalcDelta_k(*data_result_ptr);
    for (int i = 0; i < 100; ++i) {
      float test_value = sigmoid_l2_node_ptr_->get_c_delta_ptr()->array()(i);
      float correct_value = correct_delta_ptr->array()(i);
      EXPECT_NEAR(test_value, correct_value, kRelativeError_ * abs(correct_value));
    }
  }
}

// Tests SigmoidNode
