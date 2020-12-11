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
#ifndef INTELLGRAPH_SRC_TENSOR_DYN_MATRIX_H_
#define INTELLGRAPH_SRC_TENSOR_DYN_MATRIX_H_

#include <memory>

#include "glog/logging.h"
#include "src/eigen.h"

namespace intellgraph {

template <typename T> class DynMatrix {
public:
  DynMatrix();
  explicit DynMatrix(int row, int col);

  DynMatrix(const DynMatrix &) = delete;
  DynMatrix &operator=(const DynMatrix &) = delete;

  // Move constructor and assignment
  DynMatrix(DynMatrix &&matrix);
  DynMatrix &operator=(DynMatrix &&matrix);

  ~DynMatrix();

  int row() const { return row_; }

  int col() const { return col_; }

  int size() const { return size_; }

  const Eigen::Map<const MatrixX<T>> &map() const { return const_data_map_; }

  Eigen::Map<MatrixX<T>> mutable_map() { return data_map_; }

  T *data() { return data_.get(); }

  void Resize(int row, int col);

private:
  int row_ = 0;
  int col_ = 0;
  int size_ = 0;

  std::unique_ptr<T[]> data_;
  Eigen::Map<MatrixX<T>> data_map_ = Eigen::Map<MatrixX<T>>(nullptr, -1, -1);
  Eigen::Map<const MatrixX<T>> const_data_map_ =
      Eigen::Map<const MatrixX<T>>(nullptr, -1, -1);
};

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_TENSOR_DYN_MATRIX_H_
