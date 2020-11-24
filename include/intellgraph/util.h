/* Copyright 2020 The IntellGraph Authors. All Rights Reserved.
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
#ifndef INTELLGRAPH_SRC_UTILITY_UTIL_H_
#define INTELLGRAPH_SRC_UTILITY_UTIL_H_

#include <type_traits>

namespace intellgraph {

// Explicitly cast enum class to integer
template <typename Enumeration>
constexpr std::enable_if_t<std::is_enum<Enumeration>::value,
                           std::underlying_type_t<Enumeration>>
EnumToNumber(const Enumeration value) {
  return static_cast<std::underlying_type_t<Enumeration>>(value);
}

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_UTILITY_UTIL_H_
