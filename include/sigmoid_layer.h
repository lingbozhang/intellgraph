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
#ifndef NICOLE_LAYER_SIGMOID_LAYER_H_
#define NICOLE_LAYER_SIGMOID_LAYER_H_

#include <vector>

#include "layer/activation_base_layer.h"
#include "layer/layer_parameter.h"
#include "utility/common.h"

namespace nicole {
// SigmoidLayer improves performance of CallActFxn and CalcActPrime with Eigen 
// library and has better performance than ActivationLayer
template <class T>
class SigmoidLayer : public ActivationBaseLayer<T, SigmoidLayer<T>> {
 public:
  explicit SigmoidLayer(const struct LayerParameter& layer_param);

  ~SigmoidLayer() {}

  void PrintAct() const final; 

  void PrintDelta() const final;

  void PrintBias() const final;

  void CallActFxn();

  void CalcActPrime();

  inline std::vector<size_t> GetDims() {
    return layer_param_.dims;
  }

  inline MatXXSPtr<T> GetActivationPtr() {
    return activation_ptr_;
  }

  inline void SetActivationPtr(MatXXSPtr<T>& activation_ptr) {
    activation_ptr_ = activation_ptr;
  };

  inline MatXXSPtr<T> GetBiasPtr() {
    return bias_ptr_;
  }

  inline void SetBias(MatXXSPtr<T>& bias_ptr) {
    bias_ptr_ = bias_ptr;
  }

  inline MatXXSPtr<T> GetDeltaPtr() {
    return delta_ptr_;
  }

  inline void SetDeltaPtr(MatXXSPtr<T>& delta_ptr) {
    delta_ptr_ = delta_ptr;
  }

  inline bool IsActivated() {
    return activated_;
  }

 private:
  const struct LayerParameter layer_param_;
  MatXXSPtr<T> activation_ptr_;
  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXSPtr<T> delta_ptr_;
  MatXXSPtr<T> bias_ptr_;
  // True if the CallActFxn() has been called
  bool activated_;
};

// Alias for shared SigmoidLayer pointer
template <class T>
using SigLayerSPtr = std::shared_ptr<SigmoidLayer<T>>;

}  // namespace nicole

#endif  // NICOLE_LAYER_SIGMOID_LAYER_H_







