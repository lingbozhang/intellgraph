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
#ifndef INTELLGRAPH_SRC_GRAPH_CLASSIFIER_IMPL_H_
#define INTELLGRAPH_SRC_GRAPH_CLASSIFIER_IMPL_H_

#include <map>
#include <memory>
#include <set>

#include "src/edge.h"
#include "src/edge/op_vertex.h"
#include "src/edge/output_vertex.h"
#include "src/edge/vertex/input_vertex.h"
#include "src/eigen.h"
#include "src/graph.h"
#include "src/proto/graph_parameter.pb.h"
#include "src/solver.h"
#include "src/visitor.h"

namespace intellgraph {

template <typename T> class ClassifierImpl : public Graph<T> {
public:
  explicit ClassifierImpl(const GraphParameter &graph_parameter);
  ~ClassifierImpl() override;

  void Initialize(Visitor<T> &init_visitor) override;
  void Train(const MatrixX<T> &feature,
             const Eigen::Ref<const MatrixX<int>> &labels) override;
  T CalculateLoss(const MatrixX<T> &test_feature,
                  const MatrixX<int> &test_labels) override;
  void SetSolver(std::unique_ptr<Solver<T>> solver) override;

  const MatrixX<T> GetProbabilityDist(const MatrixX<T> &feature);

  // Used for threshold-moving/threshold-tuning
  // In the binary classification, predication that is greater than the
  // threshold will be classified as class 1, and 0 vice versa.
  // In the multi-class classification, predication is first updated by
  // multiplication with the threshold matrix, and the class with a maximum
  // updated predication value is selected as the predicted class.
  void SetThreshold(const MatrixX<T> &threshold);

  // Calculates and returns the Confusion Matrix based on the input data. In the
  // matrix, the row index represents a predicted class and the column index
  // represents an actual class. For example, the Confusion Matrix for the
  // binary classification is presented in the following format:
  // --------------------------------------------------------------------------
  //                       | Negative(0)       | Positive(1)
  // --------------------------------------------------------------------------
  // Predicted negative(0) | true negative     | false negative
  // Predicted positive(1) | false positive    | true positive
  // --------------------------------------------------------------------------
  const MatrixX<T>
  CalcConfusionMatrix(const MatrixX<T> &test_feature,
                      const Eigen::Ref<const MatrixX<int>> &test_labels);

private:
  void ZeroInitializeVertex();
  void Forward(const MatrixX<T> &feature);
  void Backward(const Eigen::Ref<const MatrixX<int>> &labels);

  int batch_size_ = 0;
  std::unique_ptr<Solver<T>> solver_;
  MatrixX<T> threshold_;
  InputVertex<T> *input_vertex_ = nullptr;
  OutputVertex<T> *output_vertex_ = nullptr;
  std::map<int, std::unique_ptr<OpVertex<T>>> vertex_by_id_;
  std::map<int, std::unique_ptr<Edge<T>>> edge_by_id_;
};

// Tells compiler not to instantiate the template in translation units that
// include this header file
extern template class ClassifierImpl<float>;
extern template class ClassifierImpl<double>;

} // namespace intellgraph

#endif // INTELLGRAPH_SRC_GRAPH_CLASSIFIER_IMPL_H_
