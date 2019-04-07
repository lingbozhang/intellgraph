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

#include "node/node_edge_interface.h"
#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
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
// a base interface. In Node classes, in order to save memory, only one vector 
// (activation vector) is used to store all weighted_sum, activation, and 
// activation prime results, hence, a state pattern is implemented to control 
// state transitions of the activation vector.
template <class T>
interface Node : implements NodeEdgeInterface<T> {
 public:
  // All interfaces should have virtual destructor in order to allow memory
  // release from interfaces
  virtual ~Node() noexcept = default;

  virtual void PrintAct() const = 0;

  virtual void PrintDelta() const = 0;

  virtual void PrintBias() const = 0;

  virtual bool CallActFxn() = 0;

  // Passes a functor and applies it on the activation matrix
  virtual void InitializeAct(REF const std::function<T(T)>& functor) = 0;

  virtual void InitializeBias(REF const std::function<T(T)>& functor) = 0;

  virtual void set_activation(COPY T value) = 0;

  // Get layer dimensions
  COPY virtual inline std::vector<size_t> get_dims() const = 0;

  REF virtual inline const std::vector<size_t>& ref_dims() const = 0;

  REF virtual inline const NodeParameter<T>& ref_node_param() const = 0;

  // Transitions from kAct state to kPrime state and updates current_act_state_
  virtual void ActToPrime() = 0;

  // Transitions from kInit state to kAct state and updates current_act_state_
  virtual void InitToAct() = 0;

  // Transitions from current_act_state_ to state
  virtual bool Transition(ActStates state) = 0;

};

// Alias for unique node pointer
template <class T>
using NodeUPtr = std::unique_ptr<Node<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_NODE_H_







  