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
#include <iostream>

#include "gtest/gtest.h"
#include "node/node_parameter.h"

using namespace std;
using namespace intellgraph;

// NodeParamTest contains tests for programs in node directory
class NodeParamTest : public ::testing::Test {
 protected:
  NodeParamTest() 
      : node_param2_(0, "data2", {3}), dims_({2}) {
    node_param1_ = NodeParameter<float>(0, "data1", {2});
  }
  ~NodeParamTest() {}
  NodeParameter<float> node_param1_;
  NodeParameter<float> node_param2_;
  vector<size_t> dims_;
};

TEST_F(NodeParamTest, TestAccesors) {
  EXPECT_EQ(0, node_param1_.ref_id());
  EXPECT_EQ("data1", node_param1_.ref_node_name());
  EXPECT_EQ(dims_, node_param1_.ref_dims());

  EXPECT_EQ(0, node_param1_.ref_id());
  EXPECT_EQ("data1", node_param1_.ref_node_name());
  EXPECT_EQ(dims_, node_param1_.ref_dims());
}

TEST_F(NodeParamTest, TestMutators) {
  EXPECT_EQ(10, node_param1_.set_id(10).ref_id());
  EXPECT_EQ("hello", node_param1_.set_node_name("hello").ref_node_name());
  dims_[0] = 3;
  EXPECT_EQ(dims_, node_param1_.set_dims({3}).ref_dims());
}

TEST_F(NodeParamTest, TestMoveAssignment) {
  node_param1_ = move(node_param2_);
  EXPECT_EQ("data2", node_param1_.ref_node_name());
}