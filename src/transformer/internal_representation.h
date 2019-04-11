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
#ifndef INTELLGRAPH_TRANSFORMER_INTERNAL_REPRESENTATION_H_
#define INTELLGRAPH_TRANSFORMER_INTERNAL_REPRESENTATION_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// IntRepr class is used to achieve data transfomation between different data
// types. It uses 1D array as its internal data abstraction. Note current IntRepr
// transforms data through copying, therefore has some performance penalties for
// large data structures. In the IntRepr, internal buffer will be deleted at the 
// end of conversion.
template <class T>
class IntRepr {
 public:
  explicit IntRepr(REF const std::vector<T>& vec) 
      : size_(vec.size()), row_(vec.size()), col_(1) {
    buffer_ = new T[size_];
    std::copy(vec.begin(), vec.end(), buffer_);
  }

  explicit IntRepr(REF const std::vector<std::vector<T>>& matrix) 
      : size_(matrix.size() * matrix[0].size()), row_(matrix.size()), \
        col_(matrix[0].size()) {
    buffer_ = new T[size_];
    for (size_t c = 0; c < col_; ++c) {
      for (size_t r = 0; r < row_; ++r) {
        buffer_[r + (c * row_)] = matrix[r][c];
      }
    }
  }

  explicit IntRepr(REF const MatXX<T>* matrix_ptr) 
      : size_(matrix_ptr->size()), row_(matrix_ptr->rows()), \
        col_(matrix_ptr->cols()) {
    buffer_ = new T[size_];
    for (size_t c = 0; c < col_; ++c) {
      for (size_t r = 0; r < row_; ++r) {
        buffer_[r + (c * row_)] = matrix_ptr->array()(r, c);
      }
    }    
  }

  ~IntRepr() {
   if (buffer_ != nullptr) {
     delete buffer_;
   }
  }

  COPY std::vector<T> ToStdVec(COPY size_t len) {
    if (buffer_ == nullptr || len > size_) {
      LOG(WARNING) << "ToStdVec() for IntRepr is failed, an empty "
                   << "vector is returned";
      return {};
    } else {
      delete buffer_;
      buffer_ = nullptr;
      return std::vector<T>(buffer_, buffer_ + len);
    }
  }

  COPY std::vector<std::vector<T>> ToStdMatrix(COPY size_t row, COPY size_t col) {
    std::vector<std::vector<T>> rv(row, std::vector<T>(col, 0));
    if (buffer_ == nullptr || row * col != size_) {
      LOG(WARNING) << "ToStdMatrix() for IntRepr is failed, an empty "
                   << "matrix is returned";
      return {{}};
    } else {
      for (size_t c = 0; c < col; ++c) {
        for (size_t r = 0; r < row; ++r) {
          rv[r][c] = buffer_[r + (c * row)];
        }
      }
      delete buffer_;
      buffer_ = nullptr;
      return rv;
    }
  }

  COPY MatXX<float> ToMatXXUPtr(COPY size_t row, COPY size_t col) {
    if (buffer_ == nullptr || row * col != size_) {
      LOG(WARNING) << "ToMatXXPtr() for IntRepr is failed, an empty "
                   << "nullptr is returned";
      return {};
    } else {
      MatXX<float> rv(row, col);
      for (size_t r = 0; r < row; ++r) {
        for (size_t c = 0; c < col; ++c) {
          rv(r, c) = static_cast<float>(buffer_[r + (c * row)]);
        }
      }
      delete buffer_;
      buffer_ = nullptr;
      return rv;
    }
  }

  T* buffer_{nullptr};
  size_t size_{0};
  size_t row_{0};
  size_t col_{0};
};

}  // intellgraph

#endif  // INTELLGRAPH_TRANSFORMER_INTERNAL_REPRESENTATION_H_