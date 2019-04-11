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
#ifndef INTELLGRAPH_NODE_NODE_ACTIVATOR_H_
#define INTELLGRAPH_NODE_NODE_ACTIVATOR_H_

#include <memory>

#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

template <class T>
interface Activator {
 public:
  virtual ~Activator() noexcept = default;
  
  virtual Activate(MUTE MatXX<T>* activation_ptr) = 0;

  virtual CalcActPrime(MUTE MatXX<T>* activation_ptr) = 0;

  virtual Evaluate(REF const MatXX<T>* activation_ptr, \
                   REF const MatXX<T>* labels_ptr) = 0;

};

template <class T>
using ActivatorUPtr = std::unique_ptr<Activator<T>>;

}  // intellgraph

#endif  // INTELLGRAPH_NODE_NODE_ACTIVATOR_H_
