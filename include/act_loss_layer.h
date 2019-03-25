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
#ifndef NICOLE_LAYER_ACT_LOSS_LAYER_H_
#define NICOLE_LAYER_ACT_LOSS_LAYER_H_
 
#include <vector>
// Your project's .h files
#include "layer/act_loss_base_layer.h"
#include "layer/layer_parameter.h"
#include "utility/common.h"

namespace nicole {
// ActLossLayer allows user provided function pointers. ActLossLayer 
// constructor accepts five parameters: 
// 1. layer_param: layer paramters
// 2. act_function_ptr: activation function pointer
// 3. act_prime_ptr: activation prime function pointer
// 4. loss_function_ptr: loss function pointer
// 5. loss_prime_ptr: loss function prime pointer
template <class T>
class ActLossLayer : public ActLossBaseLayer<T, ActLossLayer<T>> {
 public:
  explicit ActLossLayer(    
      const struct LayerParameter& layer_param,
      std::function<T(T)> act_function_ptr,
      std::function<T(T)> act_prime_ptr,
      std::function<T(MatXXSPtr<T>, MatXXSPtr<T>)> loss_function_ptr,
      std::function<MatXXSPtr<T>(MatXXSPtr<T>, MatXXSPtr<T>)> loss_prime_ptr);

  ~ActLossLayer() {
    std::cout << "ActLossLayer " << layer_param_.id << " is successfully deleted."
              << std::endl;
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
  
  // Note this function calls loss function at runtime and thus has performance
  // penalty
  T CalcLoss(MatXXSPtr<T>& data_result);

  // Calculates derivative of loss function of weighted_sum variables. Note this
  // function calls loss prime function at runtime and thus has performance
  // penalty
  void CalcDelta(MatXXSPtr<T>& data_result);

  inline std::vector<size_t> GetDims() {
    return layer_param_.dims;
  }

 private:
  const struct LayerParameter layer_param_;
  std::function<T(T)> act_function_ptr_;
  std::function<T(T)> act_prime_ptr_;
  std::function<T(MatXXSPtr<T>, MatXXSPtr<T>)> loss_function_ptr_;
  // Stores derivative of loss function of activation
  std::function<MatXXSPtr<T>(MatXXSPtr<T>, MatXXSPtr<T>)> loss_prime_ptr_;

  MatXXSPtr<T> activation_ptr_;
  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXSPtr<T> delta_ptr_;
  MatXXSPtr<T> bias_ptr_;
  // True if the CallActFxn() has been called
  bool activated_;
  T loss_;
};

template <class T>
using ActLossLayerSPtr = std::shared_ptr<ActLossLayer<T>>;

}  // namespace nicole

#endif  // NICOLE_LAYER_ACT_LOSS_LAYER_H_







  