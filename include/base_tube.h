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
#ifndef NICOLE_TUBE_BASE_TUBE_H_
#define NICOLE_TUBE_BASE_TUBE_H_

#include "utility/common.h"

namespace nicole {
// In Nicole, tube is a basic building block that is used to connect between 
// two layers. It is an abstract class for all tube classes and has two member 
// functions:
// 1. PrintWeight
// 2. PrintNablaWeight
class BaseTube {
 public:
  virtual void PrintWeight() const = 0;

  virtual void PrintNablaWeight() const = 0;

 protected:
  BaseTube() {}

  ~BaseTube() {}
};

// Alias for unique BaseTube pointer
using BaseTubeUPtr = std::unique_ptr<BaseTube>;

}  // namespace nicole

#endif  // NICOLE_TUBE_BASE_TUBE_H_







