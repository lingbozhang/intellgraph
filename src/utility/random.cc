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
#include "src/utility/random.h"

static std::random_device rd;
static std::mt19937 gen;

namespace intellgraph {

template <typename T>
NormalFunctor<T>::NormalFunctor(T mean, T standard_deviation) noexcept
    : mean_(mean), standard_deviation_(standard_deviation) {
  gen = std::mt19937(rd());
  nd_ = std::normal_distribution<T>(mean, standard_deviation);
}

template <typename T> T NormalFunctor<T>::operator()(const T) {
  return nd_(gen);
}

// Explicit instantiation
template class NormalFunctor<float>;
template class NormalFunctor<double>;

template <typename T>
UniformFunctor<T>::UniformFunctor(T a, T b) noexcept : a_(a), b_(b) {
  gen = std::mt19937(rd());
  dis_ = std::uniform_real_distribution<>(a, b);
}

template <typename T> T UniformFunctor<T>::operator()(const T) {
  return dis_(gen);
}

template <typename T>
BernoulliFunctor<T>::BernoulliFunctor(T a) noexcept : a_(a) {
  gen = std::mt19937(rd());
  dis_ = std::bernoulli_distribution(a);
}

template <typename T> T BernoulliFunctor<T>::operator()(const T input) {
  return input ? dis_(gen) : 0.0;
}

} // namespace intellgraph
