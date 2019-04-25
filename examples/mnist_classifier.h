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
#ifndef INTELLGRAPH_EXAMPLE_MNIST_CLASSIFIER_H_
#define INTELLGRAPH_EXAMPLE_MNIST_CLASSIFIER_H_

#include <iostream>

#include "edge/edge_headers.h"
#include "load_mnist_data.h"
#include "graph/graphfxn.h"
#include "node/node_headers.h"
#include "solver/gradient_decent.h"
#include "transformer/internal_representation.h"
#include "utility/common.h"

using namespace std;
using namespace intellgraph;
using namespace Eigen;

class Example2 {
 public:
  static void run() {
    std::cout << "=====================================" << std::endl;
    std::cout << "A Simple Classifier for MNIST dataset" << std::endl;
    std::cout << "=====================================" << std::endl;

    size_t nbr_training_data = 50000;
    size_t nbr_test_data = 10000;
    MatXX<float> training_images, training_labels;
    MatXX<float> test_images, test_labels;

    LoadMNIST<float>::LoadData(nbr_training_data,
                               nbr_test_data,
                               training_images,
                               training_labels,
                               test_images,
                               test_labels);

    // Registers node and edge classes
    NodeRegistry::LoadNodeRegistry();
    EdgeRegistry::LoadEdgeRegistry();
    // SigmoidNode is an input layer which uses Sigmoid function as activation
    // function. 
    // ReLUNode uses rectified linear unit function as the activation function
    // SigCEode uses Sigmoid function as activation function and the cross \
    // entropy function as loss function.
    Graphfxn<float> classifier;
    // Turns dropout on: dropout probability 0.5
    //classifier.TurnDropoutOn(0.5);
    classifier.AddEdge({"SigmoidNode", {0, 784}}, {"ReLUNode", {1, 100}}, \
                       "DenseEdge")
              .AddEdge({"ReLUNode", {1, 100}}, {"SigCENode", {2, 10}}, \
                       "DenseEdge")
              .Create();

    float eta = 0.1;
    float lambda = 5.0 / nbr_training_data;
    GDSolver<float> solver(eta, lambda);
    int loops = 100;
    int minbatch_size = 10;
    std::cout << "Learning rate: " << eta << std::endl;
    std::cout << "Total epochs: " << loops << std::endl;
    std::cout << "Min-batch size: " << minbatch_size << std::endl;
    // Permutation Matrix for random shuffle
    PermutationMatrix<Dynamic, Dynamic> perm(nbr_training_data);
    for (int epoch = 0; epoch < loops; ++epoch) {
      std::cout << "Epoch: " << epoch << "/" << loops <<std::endl;
      // Random shuffle
      // obtain a time-based seed:
      unsigned seed = std::chrono::system_clock::now(). \
          time_since_epoch().count();
      std::default_random_engine e(seed);
      perm.setIdentity();
      std::shuffle(perm.indices().data(), perm.indices().data() + \
          perm.indices().size(), e);
      training_images = training_images * perm;
      training_labels = training_labels * perm;
      for (int i = 0; i < nbr_training_data - minbatch_size; i += minbatch_size) {
        solver.Train(training_images.block(0, i, 784, minbatch_size), \
                     training_labels.block(0, i, 10, minbatch_size), \
                     &classifier);
      }
      classifier.Evaluate(test_images, test_labels);
    }
    std::cout << "Complete" << std::endl;
  }
};

#endif  // INTELLGRAPH_EXAMPLE_MNIST_CLASSIFIER_H