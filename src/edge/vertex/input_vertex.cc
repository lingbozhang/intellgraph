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
#include "src/edge/vertex/input_vertex.h"

#include "src/logging.h"

namespace intellgraph{

template <typename T>
InputVertex<T>::InputVertex() = default;

template <typename T>
InputVertex<T>::~InputVertex() = default;

template <typename T> void InputVertex<T>::Activate() {}

template <typename T> void InputVertex<T>::Derive() {}

template <typename T> void InputVertex<T>::ResizeVertex(int length) {}

template <typename T> Eigen::Map<MatrixX<T>> InputVertex<T>::mutable_act() {
  NOTREACHED();
  return Eigen::Map<MatrixX<T>>(nullptr, -1, -1);
}

template <typename T> Eigen::Map<MatrixX<T>> InputVertex<T>::mutable_delta() {
  return Eigen::Map<MatrixX<T>>(nullptr, -1, -1);
}

template <typename T> Eigen::Map<MatrixX<T>> InputVertex<T>::mutable_bias() {
  NOTREACHED();
  return Eigen::Map<MatrixX<T>>(nullptr, -1, -1);
}

template <typename T> const MatrixX<T> InputVertex<T>::CalcNablaBias() {
  NOTREACHED();
  return MatrixX<T>(-1, -1);
}

// Explicit instantiation
template class InputVertex<float>;
template class InputVertex<double>;

} // namespace intellgraph
