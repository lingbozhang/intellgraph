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

#include "node/node_parameter.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

// In IntellGraph, node is a basic building block that represents a neural
// network layer, all node class in /node directory should implement node class 
// as a base class. In Node classes, in order to save memory, only one 
// matrix (activation matrix) is used to store all weighted_sum, activation, and 
// activation prime results, hence, a state pattern is implemented to control 
// the state transitions of the activation matrix.
template <class T>
class Node {
 public:
  // All interfaces/abstract classes should have virtual destructor in order to 
  // release memory of derived object from interfaces/abstract classes
  virtual ~Node() noexcept = default;

  COPY virtual inline std::vector<size_t> get_dims() const = 0;

  REF virtual inline const std::vector<size_t>& ref_dims() const = 0;

  // Accessable operations for the activation matrix
  MUTE virtual inline MatXX<T>* get_activation_ptr() const = 0;

  // Accessable operations for the bias vector
  MUTE virtual inline VecX<T>* get_bias_ptr() const = 0;

  // Accessable operations for the delta matrix
  MUTE virtual inline MatXX<T>* get_delta_ptr() const = 0;

  // Accessable operations for the node parameter
  REF virtual inline const NodeParameter& ref_node_param() const = 0;

  virtual inline void set_activation(COPY T value) = 0;
  
  virtual inline void move_activation_ptr(MOVE MatXXUPtr<T> activation_ptr) = 0;
  
  // Passes a functor and applies it on the activation matrix
  virtual void InitializeAct(REF const std::function<T(T)>& functor) = 0;

  virtual inline void move_bias_ptr(MOVE VecXUPtr<T> bias_ptr) = 0;

  virtual void InitializeBias(REF const std::function<T(T)>& functor) = 0;

  virtual inline void move_delta_ptr(MOVE MatXXUPtr<T> delta_ptr) = 0;

  virtual void PrintAct() const = 0;

  virtual void PrintDelta() const = 0;

  virtual void PrintBias() const = 0;

};

// Alias for unique node pointer
template <class T>
using NodeUPtr = std::unique_ptr<Node<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_NODE_H_







  