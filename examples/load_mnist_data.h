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
#ifndef INTELLGRAPH_EXAMPLE_LOAD_MNIST_DATA_H_
#define INTELLGRAPH_EXAMPLE_LOAD_MNIST_DATA_H_

#include "mnist/mnist_reader.hpp"
#include "transformer/internal_representation.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

using namespace std;
using namespace intellgraph;
using namespace Eigen;

template <class T>
class LoadMNIST {
 public:
  ~LoadMNIST() {}

  static bool LoadData(size_t nbr_training_data, \
                       size_t nbr_test_data, \
                       MUTE MatXX<T>& training_images, \
                       MUTE MatXX<T>& training_labels, \
                       MUTE MatXX<T>& test_images, \
                       MUTE MatXX<T>& test_labels) {
    // Loads MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>( \
        MNIST_DATA_LOCATION);
    
    if (nbr_training_data > dataset.training_images.size()) {
      std::cout << "ERROR: nbr_training_data is larger than maximum dataset"
                << " size";
      return false;
    }
    if (nbr_test_data > dataset.test_images.size()) {
      std::cout << "ERROR: nbr_testdata is larger than maximum dataset"
                << " size";
      return false;
    }

    std::cout << "Nbr of training data = " << nbr_training_data << std::endl;
    std::cout << "Nbr of test data = " << nbr_test_data << std::endl;

    // Converts vector data structure to MatXX data structure
    std::vector<std::vector<uint8_t>> training_images_vec( \
        dataset.training_images.begin(),
        dataset.training_images.begin() + nbr_training_data);
    
    std::vector<std::vector<uint8_t>> training_labels_vec(nbr_training_data, \
        std::vector<uint8_t>(10, 0));
    
    std::vector<std::vector<uint8_t>> test_images_vec( \
        dataset.test_images.begin(), 
        dataset.test_images.begin() + nbr_test_data);

    std::vector<uint8_t> test_labels_vec( \
        dataset.test_labels.begin(), 
        dataset.test_labels.begin() + nbr_test_data);
    
    for (size_t i = 0; i < nbr_training_data; ++i) {
      size_t index = dataset.training_labels[i];
      training_labels_vec[i][index] = 1;
    }

    IntRepr<uint8_t> training_images_int(training_images_vec);
    IntRepr<uint8_t> training_labels_int(training_labels_vec);
    IntRepr<uint8_t> test_images_int(test_images_vec);
    IntRepr<uint8_t> test_labels_int(test_labels_vec);

    training_images = training_images_int.ToMatXXUPtr(nbr_training_data, 784);
    training_labels = training_labels_int.ToMatXXUPtr(nbr_training_data, 10);
    test_images = test_images_int.ToMatXXUPtr(nbr_test_data, 784);
    test_labels = test_labels_int.ToMatXXUPtr(nbr_test_data, 1);
  
    // Normalizes features (this is very important !)
    training_images.array() /= 255.0;
    test_images.array() /= 255.0;
    training_images.transposeInPlace();
    test_images.transposeInPlace();
    training_labels.transposeInPlace();
    test_labels.transposeInPlace();
    return true;
  }
  
};

#endif  // INTELLGRAPH_EXAMPLE_LOAD_MNIST_DATA_H_