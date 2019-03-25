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
#ifndef NICOLE_LAYER_ACTIVATION_LAYER_H_
#define NICOLE_LAYER_ACTIVATION_LAYER_H_
 
#include <vector>
// Your project's .h files
#include "layer/activation_base_layer.h"
#include "layer/layer_parameter.h"
#include "utility/common.h"

namespace nicole {
// ActivationLayer allows user provided function pointers. The constructor 
// accepts three parameters:
// 1. layer_param: layer parameter
// 2. act_function_ptr: activation function pointer
// 3. act_prime_ptr: activation prime function pointer
template <class T>
class ActivationLayer : public ActivationBaseLayer<T, ActivationLayer<T>> {
 public:
  explicit ActivationLayer(const struct LayerParameter& layer_param,
                           std::function<T(T)> act_function_ptr,
                           std::function<T(T)> act_prime_ptr);

  ~ActivationLayer() {
    std::cout << "ActivationLayer " << layer_param_.id 
              << " is successfully deleted." << std::endl;
  }

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;

  // Calls activation function and updates activation. Note this function calls 
  // activation function at runtime and thus has performance penalty
  void CallActFxn();

  // Calculates derivative of the activation function and overwrites the 
  // activation in-place. Note this function calls activation prime function at 
  // runtime and thus has performance penalty
  void CalcActPrime();

  inline std::vector<size_t> GetDims() {
    return layer_param_.dims;
  }

  inline MatXXSPtr<T> GetActivationPtr() {
    return activation_ptr_;
  }

  inline void SetActivation(MatXXSPtr<T>& activation_ptr) {
    activation_ptr_ = activation_ptr;
  };

  inline MatXXSPtr<T> GetDeltaPtr() {
    return delta_ptr_;
  }

  inline void SetDelta(MatXXSPtr<T>& delta_ptr) {
    delta_ptr_ = delta_ptr;
  }

  inline MatXXSPtr<T> GetBiasPtr() {
    return bias_ptr_;
  }

  inline void SetBias(MatXXSPtr<T>& bias_ptr) {
    bias_ptr_ = bias_ptr;
  }

  inline bool IsActivated() {
    return activated_;
  }

 private:
  const struct LayerParameter layer_param_;
  std::function<T(T)> act_function_ptr_;
  std::function<T(T)> act_prime_ptr_;

  MatXXSPtr<T> activation_ptr_;
  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXSPtr<T> delta_ptr_;
  MatXXSPtr<T> bias_ptr_;
  // True if the CallActFxn() has been called
  bool activated_;
};

template <class T>
using ActLayerSPtr = std::shared_ptr<ActivationLayer<T>>;

}  // namespace nicole

#endif  // NICOLE_LAYER_ACTIVATION_LAYER_H_







  