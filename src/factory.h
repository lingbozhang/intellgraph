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
#ifndef INTELLGRAPH_SRC_EDGE_FACTORY_H_
#define INTELLGRAPH_SRC_EDGE_FACTORY_H_

#include <functional>
#include <map>
#include <string>

#include "glog/logging.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/proto/vertex_parameter.pb.h"

namespace intellgraph {

class Factory {
public:
  template <class Base>
  using VertexConstructor =
      std::function<std::unique_ptr<Base>(const VertexParameter &, int)>;
  template <class Base>
  using VertexRegistryMap = std::map<std::string, VertexConstructor<Base>>;

  template <class Base, class VertexIn, class VertexOut>
  using EdgeConstructor =
      std::function<std::unique_ptr<Base>(int, VertexIn *, VertexOut *)>;
  template <class Base, class VertexIn, class VertexOut>
  using EdgeRegistryMap =
      std::map<std::string, EdgeConstructor<Base, VertexIn, VertexOut>>;

  template <class Base>
  using SolverConstructor =
      std::function<std::unique_ptr<Base>(const SolverConfig &)>;
  template <class Base>
  using SolverRegistryMap = std::map<std::string, SolverConstructor<Base>>;

  Factory() = delete;
  ~Factory() = delete;

  template <class Base>
  static std::unique_ptr<Base> InstantiateVertex(const VertexParameter &param,
                                                 int batch_size) {
    const std::string &type = param.type();
    if (Factory::VertexRegistry<Base>().find(type) !=
        Factory::VertexRegistry<Base>().end()) {
      return Factory::VertexRegistry<Base>().at(type)(param, batch_size);
    }
    LOG(FATAL) << "Failed to find the vertex constructor " << type
               << " from the Registry";
    return nullptr;
  }
  template <class Base> static VertexRegistryMap<Base> &VertexRegistry() {
    static VertexRegistryMap<Base> impl;
    return impl;
  }

  template <class Base, class VertexIn, class VertexOut = VertexIn>
  static std::unique_ptr<Base> InstantiateEdge(const std::string &type, int id,
                                               VertexIn *vtx_in,
                                               VertexOut *vtx_out) {
    if (Factory::EdgeRegistry<Base, VertexIn, VertexOut>().find(type) !=
        Factory::EdgeRegistry<Base, VertexIn, VertexOut>().end()) {
      return Factory::EdgeRegistry<Base, VertexIn, VertexOut>().at(type)(
          id, vtx_in, vtx_out);
    }
    LOG(FATAL) << "Failed to find the edge constructor " << type
               << " from the Registry";
    return nullptr;
  }
  template <class Base, class VertexIn, class VertexOut = VertexIn>
  static EdgeRegistryMap<Base, VertexIn, VertexOut> &EdgeRegistry() {
    static EdgeRegistryMap<Base, VertexIn, VertexOut> impl;
    return impl;
  }

  template <class Base>
  static std::unique_ptr<Base> InstantiateSolver(const SolverConfig &param) {
    const std::string &type = param.type();
    if (Factory::SolverRegistry<Base>().find(type) !=
        Factory::SolverRegistry<Base>().end()) {
      return Factory::SolverRegistry<Base>().at(type)(param);
    }
    LOG(FATAL) << "Failed to find the solver constructor " << type
               << " from the Registry";
    return nullptr;
  }
  template <class Base> static SolverRegistryMap<Base> &SolverRegistry() {
    static SolverRegistryMap<Base> impl;
    return impl;
  }
};

template <class Base, class Derived> class VertexRegister {
public:
  VertexRegister(const std::string &type) {
    Factory::VertexRegistry<Base>().try_emplace(
        type,
        [](const VertexParameter &vtx_param,
           int batch_size) -> std::unique_ptr<Base> {
          return std::make_unique<Derived>(vtx_param, batch_size);
        });
  }
};

template <class Base, class Derived, class VertexIn, class VertexOut>
class EdgeRegister {
public:
  EdgeRegister(const std::string &type) {
    Factory::EdgeRegistry<Base, VertexIn, VertexOut>().try_emplace(
        type,
        [](int id, VertexIn *vtx_in,
           VertexOut *vtx_out) -> std::unique_ptr<Base> {
          return std::make_unique<Derived>(id, vtx_in, vtx_out);
        });
  }
};

template <class Base, class Derived> class SolverRegister {
public:
  SolverRegister(const std::string &type) {
    Factory::SolverRegistry<Base>().try_emplace(
        type, [](const SolverConfig &param) -> std::unique_ptr<Base> {
          return std::make_unique<Derived>(param);
        });
  }
};

#define REGISTER_VERTEX(base, derived, type)                                   \
  static const VertexRegister<base<float>, derived<float, type>>               \
      register_f_##type##_##base##_##derived(#type);                           \
  static const VertexRegister<base<double>, derived<double, type>>             \
      register_d_##type##_##base##_##derived(#type);

#define REGISTER_EDGE(base, derived, vertex_in, vertex_out, type)              \
  static const EdgeRegister<                                                   \
      base<float>, derived<float, vertex_in<float>, vertex_out<float>>,        \
      vertex_in<float>, vertex_out<float>>                                     \
      register_f_##type##_##base##_##derived##vertex_in##vertex_out(#type);    \
  static const EdgeRegister<                                                   \
      base<double>, derived<double, vertex_in<double>, vertex_out<double>>,    \
      vertex_in<double>, vertex_out<double>>                                   \
      register_d_##type##_##base##_##derived##vertex_in##vertex_out(#type);

#define REGISTER_SOLVER(base, derived, type)                                   \
  static const SolverRegister<base<float>, derived<float>>                     \
      register_f_##type##_##base##_##derived(#type);                           \
  static const SolverRegister<base<double>, derived<double>>                   \
      register_d_##type##_##base##_##derived(#type);

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_EDGE_FACTORY_H_
