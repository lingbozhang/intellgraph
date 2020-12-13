/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-1.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
        Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#include <cstdint>

namespace intellgraph {

static inline int64_t ipow(int32_t base, uint8_t exp) {
  static const uint8_t highest_bit_set[] = {
      0,   1,   2,   2,   3,   3,   3,   3,   4,   4,   4,   4,   4,   4,   4,
      4,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      5,   5,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
      6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
      6,   6,   6,   255, // anything past 63 is a guaranteed overflow with base
                          // > 1
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 25,  255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  };

  int64_t result = 1;

  switch (highest_bit_set[exp]) {
  case 255: // we use 255 as an overflow marker and return 0 on
            // overflow/underflow
    if (base == 1) {
      return 1;
    }

    if (base == -1) {
      return 1 - 2 * (exp & 1);
    }

    return 0;
  case 6:
    if (exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  case 5:
    if (exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  case 4:
    if (exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  case 3:
    if (exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  case 2:
    if (exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  case 1:
    if (exp & 1)
      result *= base;
  default:
    return result;
  }
}

} // namespace intellgraph
