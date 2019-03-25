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
// network layer, all class in the /node directory should use node class as 
// a base class. In Node, in order to save memory, only one vector (activation 
// vector) is used to store weighted_sum, activation, and activation prime 
// results, hence, a state pattern is implemented to control state transitions 
// of the activation vector.
template <class T>
class Node {
 public:
  virtual void PrintAct() const = 0;

  virtual void PrintDelta() const = 0;

  virtual void PrintBias() const = 0;

  // This function calls activation function
  virtual void CallActFxn() = 0;

  // This function calls activation prime function 
  virtual void CalcActPrime() = 0;

  // Passes a functor and applies it on the activation vector
  virtual void ApplyUnaryFunctor(std::function<T(T)> functor) = 0;

  // Get layer dimensions
  // Note it is not a constant getter, and exposes data
  virtual inline std::vector<size_t> GetDims() = 0;

  // Note it is not a constant getter, and exposes data
  virtual inline MatXXSPtr<T> GetActivationPtr() = 0;

  virtual inline void SetActivationPtr(MatXXSPtr<T>& activation_ptr) = 0;

  virtual inline void SetActivation(T value) = 0;

  // Note it is not a constant getter, and exposes data
  virtual inline MatXXSPtr<T> GetBiasPtr() = 0;

  virtual inline void SetBiasPtr(MatXXSPtr<T>& bias_ptr) = 0;

  // Note it is not a constant getter, and exposes data
  virtual inline MatXXSPtr<T> GetDeltaPtr() = 0;

  virtual inline void SetDeltaPtr(MatXXSPtr<T>& delta_ptr) = 0;

  // Note it is not a constant getter, and exposes data
  virtual inline bool IsActivated() = 0;

 protected:
  Node() {}
  
  ~Node() {}
};

// Alias for shared node pointer
template <class T>
using NodeSPtr = std::shared_ptr<Node<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_NODE_H_







  