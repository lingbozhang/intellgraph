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
#ifndef INTELLGRAPH_EDGE_EDGE_FACTORY_H_
#define INTELLGRAPH_EDGE_EDGE_FACTORY_H_

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
template <class T>
using NodeFunctor = std::function<NodeSPtr<T>(const NodeParameter&)>;

template <class T>
using RegistryMap = std::unordered_map<std::string, NodeFunctor<T>>;

template <class T>
class NodeFactory {
 public:
  // use this to instantiate the proper Derived class
  static NodeSPtr<T> Instantiate(const std::string& name,
                                 const NodeParameter& node_param) {
    auto it = NodeFactory::Registry().find(name);
    return it == NodeFactory::Registry().end() ? \
                 nullptr : \
                 (it->second)(node_param);
  }

  static RegistryMap<T>& Registry() {
    static RegistryMap<T> impl;
    return impl;
  }

 protected:
  NodeFactory() {}

  ~NodeFactory() {}
};

template<class T, class Derived> 
class NodeFactoryRegister {
 public:
  NodeFactoryRegister(std::string name) {
    NodeFactory<T>::Registry()[name] = \
        [](const struct NodeParameter& node_param) {
          return std::make_shared<Derived>(node_param);
        };
    std::cout << "Registering Node: '" << name << "'" << std::endl;
  }
};

// Register SigmoidNode
static NodeFactoryRegister<float, SigmoidNode<float>>
    sigmoid_node_register_f("SigmoidNode_f");
static NodeFactoryRegister<double, SigmoidNode<double>>
    sigmoid_node_register_d("SigmoidNode_d");
// Register SigL2Node
static NodeFactoryRegister<float, SigL2Node<float>>
    sigmoid_l2_node_register_f("SigL2Node_f");
static NodeFactoryRegister<double, SigL2Node<double>>
    sigmoid_l2_node_register_d("SigL2Node_d");

}  // intellgraph

#endif  // INTELLGRAPH_EDGE_EDGE_FACTORY_H_