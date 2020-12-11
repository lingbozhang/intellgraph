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
#include "src/tensor/dyn_matrix.h"

namespace intellgraph{

template <typename T>
DynMatrix<T>::DynMatrix() = default;

template <typename T>
DynMatrix<T>::DynMatrix(int row, int col)
    : row_(row), col_(col), size_(row * col) {
  DCHECK_GT(row_, 0);
  DCHECK_GT(col_, 0);
  DCHECK_GT(size_, 0);
  // Allocates raw data.
  data_ = std::make_unique<T[]>(row_ * col_);
  new (&data_map_) Eigen::Map<MatrixX<T>>(data_.get(), row_, col_);
  new (&const_data_map_) Eigen::Map<const MatrixX<T>>(data_.get(), row_, col_);
  data_map_.setZero();
}

template <typename T>
DynMatrix<T>::DynMatrix(DynMatrix &&matrix)
    : row_(matrix.row()), col_(matrix.col()), size_(matrix.size()),
      data_(std::move(matrix.data_)) {
  new (&data_map_) Eigen::Map<MatrixX<T>>(data_.get(), row_, col_);
  new (&const_data_map_) Eigen::Map<const MatrixX<T>>(data_.get(), row_, col_);
}

template <typename T>
DynMatrix<T> &DynMatrix<T>::operator=(DynMatrix &&matrix) {
  row_ = matrix.row();
  col_ = matrix.col();
  size_ = matrix.size();
  data_ = std::move(matrix.data_);
  new (&data_map_) Eigen::Map<MatrixX<T>>(data_.get(), row_, col_);
  new (&const_data_map_) Eigen::Map<const MatrixX<T>>(data_.get(), row_, col_);
  return *this;
}

template <typename T> DynMatrix<T>::~DynMatrix() = default;

template <typename T> void DynMatrix<T>::Resize(int row, int col) {
  DCHECK_GT(row, 0);
  DCHECK_GT(col, 0);

  if (row_ == row && col_ == col) {
    return;
  }
  row_ = row;
  col_ = col;
  if (size_ < row_ * col_) {
    data_ = std::make_unique<T[]>(row_ * col_);
  }
  new (&data_map_) Eigen::Map<MatrixX<T>>(data_.get(), row_, col_);
  new (&const_data_map_) Eigen::Map<const MatrixX<T>>(data_.get(), row_, col_);
  data_map_.setZero();
}

// Explicit instantiation
template class DynMatrix<float>;
template class DynMatrix<double>;

} // intellgraph 
