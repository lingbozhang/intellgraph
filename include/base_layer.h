/* Copyright 2019 The Nicole Authors. All Rights Reserved.
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
#ifndef NICOLE_LAYER_BASE_LAYER_H_
#define NICOLE_LAYER_BASE_LAYER_H_
 
#include "utility/common.h"

namespace nicole {
// In Nicole, layer is a basic building block used to represent the deep neural 
// network layer. BaseLayer is an interface for all layer classes.
class BaseLayer {
 public:
  virtual void PrintAct() const = 0;

  virtual void PrintBias() const = 0;

  virtual void PrintDelta() const = 0;

 protected:
  BaseLayer() {}
  
  ~BaseLayer() {}
};

// Alias for shared BaseLayer pointer
using BaseLayerSPtr = std::shared_ptr<BaseLayer>;

}  // namespace nicole

#endif  // NICOLE_LAYER_BASE_LAYER_H_







  