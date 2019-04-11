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
#ifndef NICOLE_NODE_NODE_REGISTRY_H_
#define NICOLE_NODE_NODE_REGISTRY_H_

#include "glog/logging.h"
#include "node/node_factory.h"
#include "node/sigmoid_l2_node.h"
#include "node/sigmoid_node.h"

namespace intellgraph {

class NodeRegistry {
 public:
  NodeRegistry() = delete;

  ~NodeRegistry() = delete;

  static void LoadNodeRegistry() {
    // Registers SigmoidNode
    LOG(INFO) << "Registering SigmoidNode";
    DEVIMPL_REGISTERIMPL_NODE(SigmoidNode, IntNode);
    // Registers SigL2Node
    LOG(INFO) << "Registering SigL2Node";
    DEVIMPL_REGISTERIMPL_NODE(SigL2Node, OutputNode);

  }
  
};

}  // intellgraph

#endif  // NICOLE_NODE_NODE_REGISTRY_H_