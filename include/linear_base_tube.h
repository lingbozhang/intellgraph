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
#ifndef NICOLE_TUBE_LINEAR_BASE_TUBE_H_
#define NICOLE_TUBE_LINEAR_BASE_TUBE_H_

#include "tube/base_tube.h"
#include "utility/common.h"

namespace nicole {
// LinearBaseTube is a tube interface that provides two member functions:
// 1. Forward
// 2. Backward
template <class T, class Implementation>
class LinearBaseTube : public BaseTube {
 public:
  // Calculates weighted sum and updates activation_ptr_ of out layer. Note CRTP
  // is used
  void Forward() {
    Impl().Forward();
  }

  // Calculates nabla_weight_ and updates delta_ptr_ of in layer. Note CRTP is 
  // used
  void Backward() {
    Impl().Backward();
  }

  // Note it is not a constant getter, and exposes data
  inline MatXXSPtr<T> GetWeightPtr() {
    Impl().GetWeightPtr();
  }

 protected:
  LinearBaseTube() {}

  ~LinearBaseTube() {}

 private:
  // Returns instance of an implementation class
  Implementation& Impl() {
    return *static_cast<Implementation*>(this);
  }
};
// Alias for unique linear base tube pointer
template <class T, class Implementation>
using LinearBaseTubeUPtr = std::unique_ptr<LinearBaseTube<T, Implementation>>;

}  // namespace nicole

#endif  // NICOLE_TUBE_LINEAR_DENSE_TUBE_H_







