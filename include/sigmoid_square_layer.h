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
#ifndef NICOLE_LAYER_SIGMOID_SQUARE_LAYER_H
#define NICOLE_LAYER_SIGMOID_SQUARE_LAYER_H

#include <vector>

#include "layer/act_loss_base_layer.h"
#include "layer/layer_parameter.h"
#include "utility/common.h"

namespace nicole {
// SigSqrLayer improves performance of GetLoss, CalcDelta, GetLoss, and 
// CalcDelta with Eigen library and has better performance than OutputLayer.
// In SigSqrLayer, sigmoid function is used as a activation function and squared
// Euclidean norm is used as a loss function
template<class T>
class SigSqrLayer : public ActLossBaseLayer<T, SigSqrLayer<T>> {
 public:
  explicit SigSqrLayer(const struct LayerParameter& layer_param);

  ~SigSqrLayer() {
    std::cout << "SigSqrLayer " << layer_param_.id << " is successfully deleted."
              << std::endl;
  }

  void PrintAct() const final;

  void PrintDelta() const final;

  void PrintBias() const final;
  
  void CallActFxn();

  void CalcActPrime();

  // Uses squared Euclidean norm as a loss function
  T GetLoss(MatXXSPtr<T>& data_result);

  void CalcDelta(MatXXSPtr<T>& data_result);

  inline std::vector<size_t> GetDims() {
    return layer_param_.dims;
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
  T loss_;
};

// Alias for shared SigSqrLayer pointer
template <class T>
using SigSqrLayerSPtr = std::shared_ptr<SigSqrLayer<T>>;

}  // namespace nicole

#endif  //NICOLE_LAYER_SIGMOID_SQUARE_LAYER_H
