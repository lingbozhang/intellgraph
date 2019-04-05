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
#ifndef INTELLGRAPH_NODE_NODE_EDGE_INTERFACE_H_
#define INTELLGRAPH_NODE_NODE_EDGE_INTERFACE_H_

#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

template <class T>
interface NodeEdgeInterface {
 public:
  virtual ~NodeEdgeInterface() noexcept = default;

  virtual void CalcActPrime() = 0;

  MUTE virtual inline MatXX<T>* get_activation_ptr() const = 0;

  virtual inline void move_activation_ptr(MOVE MatXXUPtr<T> activation_ptr) = 0;

  MUTE virtual inline MatXX<T>* get_bias_ptr() const = 0;

  virtual inline void move_bias_ptr(MOVE MatXXUPtr<T> bias_ptr) = 0;

  MUTE virtual inline MatXX<T>* get_delta_ptr() const = 0;

  virtual inline void move_delta_ptr(MOVE MatXXUPtr<T> delta_ptr) = 0;

};

}  // intellgraph

#endif  // INTELLGRAPH_NODE_NODE_EDGE_INTERFACE_H_