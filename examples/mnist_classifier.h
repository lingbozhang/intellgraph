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
#include "graph/classifier.h"
#include "graph/graph.h"
#include "mnist/mnist_reader.hpp"
#include "node/node_headers.h"
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
    // Loads MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>( \
        MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() \
              << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() \
              << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() \
              << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() \
              << std::endl;

    size_t nbr_training_data = dataset.training_images.size();
    size_t nbr_training_feature = dataset.training_images[0].size();
    size_t nbr_test_data = dataset.test_images.size();
    size_t nbr_test_feature = dataset.test_images[0].size();
    // Converts vector data structure to MatXX data structure
    std::vector<std::vector<uint8_t>> temp_labels(nbr_training_data, \
        std::vector<uint8_t>(10, 0));
    for (size_t i = 0; i < nbr_training_data; ++i) {
      size_t index = dataset.training_labels[i];
      temp_labels[i][index] = 1;
    }

    IntRepr<uint8_t> training_images_int(dataset.training_images);
    IntRepr<uint8_t> training_labels_int(temp_labels);
    IntRepr<uint8_t> test_images_int(dataset.test_images);
    IntRepr<uint8_t> test_labels_int(dataset.test_labels);

    MatXX<float> training_images = training_images_int.ToMatXXUPtr( \
        nbr_training_data, nbr_training_feature);
    MatXX<float> training_labels = training_labels_int.ToMatXXUPtr( \
        nbr_training_data, 10);
    MatXX<float> test_images = test_images_int.ToMatXXUPtr( \
        nbr_test_data, nbr_test_feature);
    MatXX<float> test_labels = test_labels_int.ToMatXXUPtr( \
        nbr_test_data, 1);
  
    // Normalizes features (this is very important !)
    training_images.array() /= 255.0;
    test_images.array() /= 255.0;
    training_images.transposeInPlace();
    test_images.transposeInPlace();
    training_labels.transposeInPlace();
    test_labels.transposeInPlace();

    // SigmoidNode is an input layer which uses Sigmoid function as activation
    // function. 
    auto node_param1 = NodeParameter(0, "SigmoidNode", {784});
    // SigmoidNode uses Sigmoid function as the activation function
    auto node_param2 = NodeParameter(1, "SigmoidNode", {30});
    // SigL2Node uses Sigmoid function as activation function and l2 norm as
    // loss function.
    auto node_param3 = NodeParameter(4, "SigL2Node", {10});

    // IntellGraph implements Boost Graph library and stores node and edge
    // information in the adjacency list.
    Classifier<float> classifier;
    // DenseEdge represents fully connected networks
    classifier.AddEdge(node_param1, node_param2, "DenseEdge");
    classifier.AddEdge(node_param2, node_param3, "DenseEdge");

    classifier.set_input_node_id(0);
    classifier.set_output_node_id(4);

    NodeRegistry::LoadNodeRegistry();
    EdgeRegistry::LoadEdgeRegistry();

    classifier.Instantiate();

    float eta = 3.0;
    int loops = 30;
    int minbatch_size = 100;
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
        classifier.Backward(training_images.block(0, i, 784, minbatch_size), \
                            training_labels.block(0, i, 10, minbatch_size));
        // Stochastic gradient decent
        classifier.get_edge_weight_ptr(1, 4)->array() -= \
            eta * classifier.get_edge_nabla_ptr(1, 4)->array();
        classifier.get_edge_weight_ptr(0, 1)->array() -= \
            eta * classifier.get_edge_nabla_ptr(0, 1)->array();
        classifier.get_node_bias_ptr(4)->array() -= eta / minbatch_size *\
            classifier.get_node_delta_ptr(4)->rowwise().sum().array();
        classifier.get_node_bias_ptr(1)->array() -= eta / minbatch_size *\
            classifier.get_node_delta_ptr(1)->rowwise().sum().array();
      }
      classifier.Evaluate(test_images, test_labels);
    }
    std::cout << "Complete" << std::endl;
  }
};

#endif  // INTELLGRAPH_EXAMPLE_MNIST_CLASSIFIER_H