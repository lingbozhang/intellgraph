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
#ifndef INTELLGRAPH_NODE_NODE_H_
#define INTELLGRAPH_NODE_NODE_H_
 
#include <functional>
#include <memory>
#include <vector>

#include "node/node_factory.h"
#include "utility/common.h"

namespace intellgraph {
// Activation vector has three states and are used to determine behaviors of 
// node methods.
enum ActStates {
  // Activation vector is initialized or overwritten with weighted_sum
  kInit = 0, 
  // CallActFxn() is invoked and activation vector is overwritten with 
  // activation values.
  kAct = 1,
  // CalcPrime() is invoked and activation vector is overwritten with prime
  // values.
  kPrime = 3,
};
// In IntellGraph, node is a basic building block that represents a neural
// network layer, all node class in /node directory should use node class as 
// a base class. In Node classes, in order to save memory, only one vector 
// (activation vector) is used to store all weighted_sum, activation, and 
// activation prime results, hence, a state pattern is implemented to control 
// state transitions of the activation vector.
template <class T>
class Node {
 public:
  Node() noexcept = default;

  virtual ~Node() noexcept = default;

  virtual void PrintAct() const = 0;

  virtual void PrintDelta() const = 0;

  virtual void PrintBias() const = 0;

  virtual void CallActFxn() = 0;

  virtual void CalcActPrime() = 0;

  // Passes a functor and applies it on the activation vector
  virtual void ApplyUnaryFunctor_k(const std::function<T(T)>& functor) = 0;

  // Get layer dimensions
  virtual inline std::vector<size_t> get_c_dims() const = 0;

  virtual inline const std::vector<size_t>& get_k_dims() const = 0;

  virtual inline MatXX<T>* get_c_activation_ptr() const = 0;

  // Setters named with letter 'm' indicates a move setter (which means 
  // argument ownerships are moved into the function)
  virtual inline void set_m_activation_ptr(MatXXUPtr<T> activation_ptr) = 0;

  virtual inline void set_c_activation(T value) = 0;

  virtual inline MatXX<T>* get_c_bias_ptr() const = 0;

  virtual inline void set_m_bias_ptr(MatXXUPtr<T> bias_ptr) = 0;

  virtual inline MatXX<T>* get_c_delta_ptr() const = 0;

  virtual inline void set_m_delta_ptr(MatXXUPtr<T> delta_ptr) = 0;

};

// Alias for unique node pointer
template <class T>
using NodeUPtr = std::unique_ptr<Node<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_NODE_H_







  