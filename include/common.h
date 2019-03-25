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
#ifndef NICOLE_UTILITY_COMMON_H_
#define NICOLE_UTILITY_COMMON_H_

#include <iostream>
#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "utility/nl_random.h"

namespace nicole {
// Alias for 2-D dynamic matrix from Eigen
template<class T>
using MatXX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
// Alias 2-D dynamic matrix unique pointer from Eigen
template<class T>
using MatXXUPtr = std::unique_ptr<MatXX<T>>;
// Alias 2-D dynamic matrix shared pointer from Eigen
template<class T>
using MatXXSPtr = std::shared_ptr<MatXX<T>>;

MatXX<double> ArrayInitSDd(size_t row, size_t col);

MatXX<float> ArrayInitSDf(size_t row, size_t col);

}  // namespace nicole

#endif //NICOLE_UTILITY_COMMON_H_



