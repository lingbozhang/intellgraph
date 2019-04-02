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
#include <iostream>
#include <unordered_map>

#include "node/node_parameter.h"

namespace intellgraph {
// A Factory design pattern, NodeFactory is used to instantiate corresponding
// node object.
template <class T, class Base>
using NodeFunctor = std::function<std::unique_ptr<Base>(const NodeParameter<T>&)>;

template <class T, class Base>
using NodeRegistryMap = std::unordered_map<std::string, NodeFunctor<T, Base>>;

template <class T, class Base>
class NodeFactory {
 public:
  NodeFactory() = delete;

  ~NodeFactory() = delete;

  // use this to instantiate the proper Derived class
  // static functions have no this parameter. They need no cv-qualifiers.
  static std::unique_ptr<Base> Instantiate(const NodeParameter<T>& node_param) {
    std::string name = node_param.get_c_node_name();
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

};

template <class T, class Base, class Derived>
class NodeFactoryRegister {
 public:
  explicit NodeFactoryRegister(const std::string& name) {
    NodeFactory<T, Base>::Registry()[name] = \
        [](const NodeParameter<T>& node_param) -> std::unique_ptr<Base> {
            // (C++14 feature)
          std::unique_ptr<Base> rv = std::make_unique<Derived>(node_param);
          return rv;
        };
    std::cout << "Registering Node: '" << name << "'" << std::endl;
  }
};

namespace devimpl {
// Registers macros are defined here, developers can use them to register their
// own node classes. Note in order to activate register macros, user defined
// classes should be added as a dynamic library (see force static variable
// initialization)
#define DEVIMPL_REGISTER(classname, base) \
  static const NodeFactoryRegister<float, base<float>, classname<float>> \
      register_f_##classname; \
  static const NodeFactoryRegister<double, base<double>, classname<double>> \
      register_d_##classname;

#define DEVIMPL_REGISTERIMPL(classname, base) \
  static const NodeFactoryRegister<float, base<float>, classname<float>> \
      register_f_##classname(#classname); \
  static const NodeFactoryRegister<double, base<double>, classname<double>> \
      register_d_##classname(#classname);

}  // devimpl

}  // intellgraph

#endif  // INTELLGRAPH_NODE_NODE_FACTORY_H_