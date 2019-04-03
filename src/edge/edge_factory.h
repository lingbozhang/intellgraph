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
#include <iostream>
#include <unordered_map>

#include "edge/edge_parameter.h"

namespace intellgraph {
// A Factory design pattern, EdgeFactory is used to instantiate corresponding
// edge object.
template <class T, class Base>
using EdgeFunctor = std::function<std::unique_ptr<Base>(const EdgeParameter&)>;

template <class T, class Base>
using EdgeRegistryMap = std::unordered_map<std::string, EdgeFunctor<T, Base>>;

template <class T, class Base>
class EdgeFactory {
 public:
  EdgeFactory() = delete;

  ~EdgeFactory()= delete;

  // use this to instantiate the proper Derived class
  static std::unique_ptr<Base> Instantiate(const EdgeParameter& edge_param) {
    std::string name = edge_param.get_k_edge_name();
    auto it = EdgeFactory::Registry().find(name);
    if (it == EdgeFactory::Registry().end() ) {
      std::cout << "WARNING: instantiate Edge " << name << " failed"
                << std::endl;
      return nullptr;
    } else {
      return (it->second)(edge_param);
    }
  }

  static EdgeRegistryMap<T, Base>& Registry() {
    static EdgeRegistryMap<T, Base> impl;
    return impl;
  }

};

template<class T, class Base, class Derived>
class EdgeFactoryRegister {
 public:
  explicit EdgeFactoryRegister(const std::string& name) {
    EdgeFactory<T, Base>::Registry()[name] = \
        [](const EdgeParameter& edge_param) -> std::unique_ptr<Base> {
          std::unique_ptr<Base> rv = std::make_unique<Derived>(edge_param);
          return rv;
        };
    //std::cout << "Registering Edge: '" << name << "'" << std::endl;
  }
};

namespace devimpl {
// Registers macros are defined here, developers can use them to register their
// own edge classes. Note in order to activate register macros, user defined
// classes should be added as a dynamic library (see force static variable
// initialization)
#define DEVIMPL_REGISTER_EDGE(classname, base) \
  static const EdgeFactoryRegister<float, base<float>, classname<float>> \
      register_f_##classname##_##base; \
  static const EdgeFactoryRegister<double, base<double>, classname<double>> \
      register_d_##classname##_##base;

#define DEVIMPL_REGISTERIMPL_EDGE(classname, base) \
  static const EdgeFactoryRegister<float, base<float>, classname<float>> \
      register_f_##classname##_##base(#classname); \
  static const EdgeFactoryRegister<double, base<double>, classname<double>> \
      register_d_##classname##_##base(#classname);

}  // devimp

}  // intellgraph

#endif  // INTELLGRAPH_EDGE_EDGE_FACTORY_H_