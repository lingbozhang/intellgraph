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
#ifndef INTELLGRAPH_UTILITY_RANDOM_H
#define INTELLGRAPH_UTILITY_RANDOM_H

#include <functional>
#include <iostream>
#include <random>

namespace intellgraph {

// Normal distribution functor
template <typename T> class NormalFunctor {
public:
  explicit NormalFunctor(T mean, T standard_deviation) noexcept;

  // Operator returns normal distribution result.
  T operator()(const T);

private:
  T mean_ = 0.0;
  T standard_deviation_ = 0.0;
  std::normal_distribution<T> nd_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class NormalFunctor<float>;
extern template class NormalFunctor<double>;

// Uniform distribution functor
template <typename T> class UniformFunctor {
public:
  explicit UniformFunctor(T a, T b) noexcept;

  // Operator returns uniform distribution result.
  T operator()(const T);

private:
  T a_ = 0.0;
  T b_ = 0.0;
  std::uniform_real_distribution<> dis_;
};

// Bernoulli distribution functor
template <class T> class BernoulliFunctor {
public:
  explicit BernoulliFunctor(T a) noexcept;

  // Operator returns Bernoulli distribution result.
  T operator()(const T input);

private:
  T a_ = 0.0;
  std::bernoulli_distribution dis_;
};

} // namespace intellgraph

#endif // INTELLGRAPH_UTILITY_RANDOM_H
