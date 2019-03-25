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
#ifndef NICOLE_LAYER_ACT_LOSS_BASE_LAYER_H_
#define NICOLE_LAYER_ACT_LOSS_BASE_LAYER_H_
 
#include <vector>

#include "layer/base_layer.h"
#include "utility/common.h"

namespace nicole {
// ActLossBaseLayer is a layer interface that provide activaton and loss methods.
// To improve its performance Curiously Recurring Template Pattern (CRTP) design 
// pattern is used.
template <class T, class Implementation>
class ActLossBaseLayer : public BaseLayer {
 public:

  // This function calls activation function with Curiously Recurring Template 
  // Pattern (CRTP) to improve the runtime performance
  void CallActFxn() {
    Impl().CallActFxn();
  }

  // This function calls activation prime function with Curiously Recurring 
  // Template Pattern (CRTP) to improve the runtime performance
  void CalcActPrime() {
    Impl().CalcActPrime();
  }

  T CalcLoss(MatXXSPtr<T>& data_result) {
    return Impl().GetLoss(data_result);
  }

  void CalcDelta(MatXXSPtr<T>& data_result) {
    Impl().CalcDelta(data_result);
  }

  // Get layer dimensions
  // Note it is not a constant getter, and exposes data
  inline std::vector<size_t> GetDims() {
    return Impl().GetDims();
  }

 protected:
  ActLossBaseLayer() {}

  ~ActLossBaseLayer() {}

 private:
  // Returns instance of an implementation class
  Implementation& Impl() {
    return *static_cast<Implementation*>(this);
  }
};

template <class T, class Implementation>
using ActLossBaseSPtr = 
    std::shared_ptr<ActLossBaseLayer<T, Implementation>>;

}  // namespace nicole

#endif  // NICOLE_LAYER_ACT_LOSS_BASE_LAYER_H_







  