//
// Created by Lingbo Zhang on 3/11/19.
//
#include "utility/common.h"

namespace intellgraph {
 MatXX<double> ArrayInitSDd(size_t row, size_t col) {
    return MatXX<double>::Zero(row, col).unaryExpr(std::function<double(double)>(standard_normald));
 }

 MatXX<float> ArrayInitSDf(size_t row, size_t col) {
    return MatXX<float>::Zero(row, col).unaryExpr(std::function<float(float)>(standard_normalf));
 }
} // namespace intellgraph