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
#ifndef NICOLE_LAYER_ACTIVATION_BASE_LAYER_H_
#define NICOLE_LAYER_ACTIVATION_BASE_LAYER_H_
 
#include <vector>
// Other libraries' .h files
// Your project's .h files
#include "layer/base_layer.h"
#include "utility/common.h"

namespace nicole {
// ActivationBaseLayer is a layer interface that provides activation methods. 
// To improve its performance, Curiously Recurring Template Pattern (CRTP) 
// design pattern is used.
template <class T, class Implementation>
class ActivationBaseLayer : public BaseLayer {
 public:

  // This function calls activation function with Curiously Recurring Template 
  // Pattern (CRTP)
  void CallActFxn() {
    Impl().CallActFxn();
  }

  // This function calls activation prime function with Curiously Recurring 
  // Template Pattern (CRTP)
  void CalcActPrime() {
    Impl().CalcActPrime();
  }

  // Get layer dimensions
  // Note it is not a constant getter, and exposes data
  inline std::vector<size_t> GetDims() {
    return Impl().GetDims();
  }

  // Note it is not a constant getter, and exposes data
  inline MatXXSPtr<T> GetActivationPtr() {
    return Impl().GetActivationPtr();
  }

  inline void SetActivationPtr(MatXXSPtr<T>& activation_ptr) {
    Impl().SetActivationPtr(activation_ptr);
  }

  // Note it is not a constant getter, and exposes data
  inline MatXXSPtr<T> GetBiasPtr() {
    return Impl().GetBiasPtr();
  }

  inline void SetBiasPtr(MatXXSPtr<T>& bias_ptr) {
    Impl().SetBiasPtr(bias_ptr);
  }

  // Note it is not a constant getter, and exposes data
  inline MatXXSPtr<T> GetDeltaPtr() {
    return Impl().GetDeltaPtr();
  }

  inline void SetDeltaPtr(MatXXSPtr<T>& delta_ptr) {
    Impl().SetDeltaPtr(delta_ptr);
  }

  // Note it is not a constant getter, and exposes data
  inline bool IsActivated() {
    return Impl().IsActivated();
  }

 protected:
  ActivationBaseLayer() {}

  ~ActivationBaseLayer() {}
 private:
  // Returns instance of an implementation class
  Implementation& Impl() {
    return *static_cast<Implementation*>(this);
  }
};

template <class T, class Implementation>
using ActBaseLayerSPtr =
    std::shared_ptr<ActivationBaseLayer<T, Implementation>>;

}  // namespace nicole

#endif  // NICOLE_LAYER_ACTIVATION_BASE_LAYER_H_







  