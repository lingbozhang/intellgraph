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
#ifndef INTELLGRAPH_EDGE_EDGE_H_
#define INTELLGRAPH_EDGE_EDGE_H_

#include <functional>

#include "edge/edge_factory.h"
#include "node/node.h"
#include "utility/common.h"

namespace intellgraph {
// In IntellGraph, edge is a basic building block that is used to connect between 
// two nodes. It is an abstract class for all edge classes and has four member 
// functions
template <class T>
class Edge {
 public:
  Edge() noexcept = default;

  virtual ~Edge() noexcept = default;

  virtual void PrintWeight() const = 0;

  virtual void PrintNablaWeight() const = 0;

  // Calculates weighted sum and updates activation_ptr_ of output layer
  // in-place. Function name with a word 'mute' indicates it requires mutable
  // inputs;
  virtual void Forward_mute(Node<T>& node_in, Node<T>& node_out) = 0;

  // Calculates nabla_weight_ and updates delta_ptr_ of input layer in-place 
  // with backpropagation
  virtual void Backward_mute(Node<T>& node_in, Node<T>& node_out) = 0;

  // Passes a unary functor and applies it on the weight matrix
  virtual void ApplyUnaryFunctor_k(const std::function<T(T)>& functor) = 0;

  virtual inline MatXX<T>* get_c_weight_ptr() const = 0;

  virtual inline MatXX<T>* get_c_nabla_weight_ptr() const = 0;

};

// Alias for unique Edge pointer
template <class T>
using EdgeUPtr = std::unique_ptr<Edge<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_EDGE_EDGE_H_







