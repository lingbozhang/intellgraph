/* Copyright 2019 The IntellGraph Authors. All Rights Reserved.
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
#ifndef INTELLGRAPH_NODE_NODE_FACTORY_H_
#define INTELLGRAPH_NODE_NODE_FACTORY_H_

#include <functional>
#include <unordered_map>

#include "node/activation_node.h"
#include "node/act_loss_node.h"
#include "node/node.h"
#include "node_parameter.h"
#include "node/sigmoid_node.h"
#include "node/sigmoid_l2_node.h"
#include "utility/common.h"

namespace intellgraph {
// A Factory design pattern, NodeFactory is used to instantiate corresponding
// node object.
template <class T, class Base>
using NodeFunctor = std::function<Base(NodeParameter)>;

template <class T, class Base>
using NodeRegistryMap = std::unordered_map<const std::string&, NodeFunctor<T, Base>>;

template <class T, class Base>
class NodeFactory {
 public:
  // use this to instantiate the proper Derived class
  static Base Instantiate(const NodeParameter& node_param) {
    const std::string& name = node_param.get_k_node_name();
    auto it = NodeFactory::Registry().find(name);
    if (it == NodeFactory::Registry().end()) {
      std::cout << "WARNING: instantiate node " << name << " failed" 
                << std::endl;
      return nullptr;
    } else {
      return (it->second)(node_param);
    }
  }

  static NodeRegistryMap<T, Base>& Registry() {
    static NodeRegistryMap<T, Base> impl;
    return impl;
  }

 protected:
  NodeFactory() {}

  ~NodeFactory() {}
};

template<class T, class Base, class Derived> 
class NodeFactoryRegister {
 public:
  NodeFactoryRegister(std::string name) {
    NodeFactory<T, Base>::Registry()[name] = \
        [](const NodeParameter& node_param) {
          return std::make_unique<Derived>(node_param);
        };
    std::cout << "Registering Node: '" << name << "'" << std::endl;
  }
};

// Register SigmoidNode
static NodeFactoryRegister<float, NodeUPtr<float>, SigmoidNode<float>>
    sigmoid_node_register_f("SigmoidNode_f");
static NodeFactoryRegister<double, NodeUPtr<double>, SigmoidNode<double>>
    sigmoid_node_register_d("SigmoidNode_d");
// Register SigL2Node
static NodeFactoryRegister<float, OutputNodeUPtr<float>, SigL2Node<float>>
    sigmoid_l2_node_register_f("SigL2Node_f");
static NodeFactoryRegister<double, OutputNodeUPtr<double>, SigL2Node<double>>
    sigmoid_l2_node_register_d("SigL2Node_d");

}  // intellgraph

#endif  // INTELLGRAPH_NODE_NODE_FACTORY_H_