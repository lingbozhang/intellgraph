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
#ifndef INTELLGRAPH_UTILITY_RANDOM_H
#define INTELLGRAPH_UTILITY_RANDOM_H

#include <functional>
#include <iostream>
#include <random>

namespace intellgraph {
static std::random_device rd;
static std::mt19937 gen;

// Normal distribution functor
template <class T>
class NormalFunctor {
 public:
  explicit NormalFunctor(T mean, T standard_deviation)
      : mean_(mean), standard_deviation_(standard_deviation) {
    gen = std::mt19937(rd());
    nd_ = std::normal_distribution<T>(mean, standard_deviation);
  }
  // Operator takes dummy argument and returns normal distribution result.
  T operator () (const T dummy) {
    return nd_(gen);
  }

 private:
  T mean_;
  T standard_deviation_;
  std::normal_distribution<T> nd_;
};

template <class T>
class UniformFunctor {
 public:
  UniformFunctor(T a, T b) : a_(a), b_(b) {
    gen = std::mt19937(rd());
    dis_ = std::uniform_real_distribution<>(a,b);
  }
  T operator () (const T dummy) {
    return dis_(gen);
  }
 private:
  T a_;
  T b_;
  std::uniform_real_distribution<> dis_;
};

}  // intellgraph

#endif // INTELLGRAPH_UTILITY_RANDOM_H