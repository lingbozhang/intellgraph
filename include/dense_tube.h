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
#ifndef NICOLE_TUBE_DENSE_TUBE_H_
#define NICOLE_TUBE_DENSE_TUBE_H_

#include "layer/activation_base_layer.h"
#include "tube/linear_base_tube.h"
#include "tube/tube_parameter.h"
#include "utility/common.h"

namespace nicole {
// DenseTube is a tube class that used to build fully connected neural networks.
// In DenseTube, weight is updated based on the backpropagation. 
template <class T, class InType, class OutType>
class DenseTube : public LinearBaseTube<T, DenseTube<T, InType, OutType>> {
 public:
  explicit DenseTube(
      const struct TubeParameter<T, InType, OutType>& tube_param);

  ~DenseTube() {
    std::cout << "DenseTube " << tube_param_.id << " is successfully deleted."
              << std::endl;
  };

  void PrintWeight() const final;

  void PrintNablaWeight() const final;

  // Calculates weighted sum and updates activation_ptr_ of output layer
  // in-place
  void Forward();

  // Calculates nabla_weight_ and updates delta_ptr_ of input layer in-place 
  // with backpropagation
  void Backward();

  inline MatXXSPtr<T> GetWeightPtr() {
    return weight_ptr_;
  }

 private:
  const struct TubeParameter<T, InType, OutType> tube_param_;
  ActBaseLayerSPtr<T, InType> in_layer_ptr_;
  ActBaseLayerSPtr<T, OutType> out_layer_ptr_;

  MatXXSPtr<T> weight_ptr_;
  MatXXSPtr<T> nabla_weight_ptr_;
};
// Alias for unique dense tube pointer
template <class T, class InType, class OutType>
using DenseTubeUPtr = std::unique_ptr<DenseTube<T, InType, OutType>>;

}  // namespace nicole

#endif  // NICOLE_TUBE_DENSE_TUBE_H_







