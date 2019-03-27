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

#include "edge/dense_edge.h"
#include "edge/edge.h"
#include "edge/edge_parameter.h"
#include "utility/common.h"

namespace intellgraph {
// A Factory design pattern, EdgeFactory is used to instantiate corresponding
// edge object.
template <class T>
using EdgeFunctor = std::function<EdgeSPtr<T>(const EdgeParameter<T>&)>;

template <class T>
using EdgeRegistryMap = std::unordered_map<std::string, EdgeFunctor<T>>;

template <class T>
class EdgeFactory {
 public:
  // use this to instantiate the proper Derived class
  static EdgeSPtr<T> Instantiate(const std::string& name,
                             const EdgeParameter<T>& edge_param) {
    auto it = EdgeFactory::Registry().find(name);
    return it == EdgeFactory::Registry().end() ? nullptr : \
                                                 (it->second)(edge_param);
  }

  static EdgeRegistryMap<T>& Registry() {
    static EdgeRegistryMap<T> impl;
    return impl;
  }

 protected:
  EdgeFactory() {}

  ~EdgeFactory() {}
};

template<class T, class Derived> 
class EdgeFactoryRegister {
 public:
  EdgeFactoryRegister(std::string name) {
    EdgeFactory<T>::Registry()[name] = \
        [](const struct EdgeParameter<T>& edge_param) {
          return std::make_shared<Derived>(edge_param);
        };
    std::cout << "Registering Edge: '" << name << "'" << std::endl;
  }
};

// Register DenseEdge
static EdgeFactoryRegister<float, DenseEdge<float>>
    dense_edge_register_f("DenseEdge_f");
static EdgeFactoryRegister<double, DenseEdge<double>>
    dense_edge_register_d("DenseEdge_d");

}  // intellgraph

#endif  // INTELLGRAPH_EDGE_EDGE_FACTORY_H_