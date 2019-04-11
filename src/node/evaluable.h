/* Copyright 2019 The IntellGraph Authors. All Rights Reserved.
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
#ifndef INTELLGRPAH_NODE_EVALUABLE_H_
#define INTELLGRPAH_NODE_EVALUABLE_H_

#include <memory>

#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

template <class T>
interface evaluable {
 public:
  virtual ~evaluable() noexcept = default;

  COPY virtual T CalcLoss(REF const Eigen::Ref<const MatXX<T>>& labels) = 0;

  virtual bool CalcDelta(REF const Eigen::Ref<const MatXX<T>>& labels) = 0;

};

}  // intellgraph

#endif  // INTELLGRPAH_NODE_EVALUABLE_H_