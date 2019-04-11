
#include <algorithm>

#include "glog/logging.h"
#include "transformer/internal_representation.h"
#include "mnist/mnist_reader.hpp"
#include "simple_classifier.h"
#include "node/node_headers.h"
#include "utility/random.h"
#include "utility/system.h"

using namespace std;
using namespace intellgraph;
using namespace Eigen;

int main(int argc, char* argv[]) {
  // Initialize Google's logging library.
  FLAGS_alsologtostderr = false;
  FLAGS_minloglevel = 2;
  std::string log_path(GetCWD());
  if (log_path.empty()) {
    // Stores log files in tmp/ if GetCWD() fails
    log_path = "tmp/";
  } else {
    log_path += "/logs";
  }

  fLS::FLAGS_log_dir = log_path;
  google::InitGoogleLogging(argv[0]);

/*
  // MNIST_DATA_LOCATION set by MNIST cmake config
  std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
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
    //cout << index << endl;
    temp_labels[i][index] = 1;
  }
  IntRepr<uint8_t> training_images(dataset.training_images);
  IntRepr<uint8_t> training_labels(temp_labels);
  IntRepr<uint8_t> test_images(dataset.test_images);
  IntRepr<uint8_t> test_labels(dataset.test_labels);

  MatXXSPtr<float> training_images_ptr = training_images.ToMatXXUPtr( \
      nbr_training_data, nbr_training_feature);
  MatXXSPtr<float> training_labels_ptr = training_labels.ToMatXXUPtr( \
      nbr_training_data, 10);
  MatXXSPtr<float> test_images_ptr = test_images.ToMatXXUPtr( \
      nbr_test_data, nbr_test_feature);
  MatXXSPtr<float> test_labels_ptr = test_labels.ToMatXXUPtr( \
      nbr_test_data, 1);
  // Normalizes features (this is very important !)
  training_images_ptr->array() /= 255.0;
  test_images_ptr->array() /= 255.0;
  training_images_ptr->transposeInPlace();
  test_images_ptr->transposeInPlace();
  training_labels_ptr->transposeInPlace();
  test_labels_ptr->transposeInPlace();

  // SigmoidNode is an input layer which uses Sigmoid function as activation
  // function. Note Node has two dimensions, the first dimension indicates
  // number of nodes and the second dimension is currently used for batch
  // size
  auto node_param1 = NodeParameter(0, "SigmoidNode", {784, 1});
  // SigmoidNode uses Sigmoid function as the activation function
  auto node_param2 = NodeParameter(1, "SigmoidNode", {30, 1});
  // SigL2Node uses Sigmoid function as activation function and l2 norm as
  // loss function.
  auto node_param3 = NodeParameter(2, "SigL2Node", {10, 1});

  // IntellGraph implements Boost Graph library and stores node and edge
  // information in the adjacency list.
  Classifier<float> classifier;
  // DenseEdge represents fully connected networks
  classifier.AddEdge(node_param1, node_param2, "DenseEdge");
  classifier.AddEdge(node_param2, node_param3, "DenseEdge");

  classifier.set_input_node_id(0);
  classifier.set_output_node_id(2);

  NodeRegistry::LoadNodeRegistry();
  EdgeRegistry::LoadEdgeRegistry();

  classifier.Instantiate();

  MatXXSPtr<float> train_data_ptr = std::make_shared<MatXX<float>>(784, 1);
  MatXXSPtr<float> train_label_ptr = std::make_shared<MatXX<float>>(10, 1);

  float eta = 3.0;
  int loops = 30;
  int mini_batich_size = 1;

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
    training_images_ptr->matrix() = training_images_ptr->matrix() * perm;
    training_labels_ptr->matrix() = training_labels_ptr->matrix() * perm;
    for (int i = 0; i < nbr_training_data; ++i) {
      train_data_ptr->array() = training_images_ptr->col(i);
      train_label_ptr->array() = training_labels_ptr->col(i);
      classifier.Backward(train_data_ptr, train_label_ptr);
      // Stochastic gradient decent
      classifier.get_edge_weight(1, 2)->array() -= \
      eta * classifier.get_edge_nabla(1, 2)->array();
      classifier.get_edge_weight(0, 1)->array() -= \
      eta * classifier.get_edge_nabla(0, 1)->array();
      classifier.get_node_bias(2)->array() -= eta * \
      classifier.get_node_delta(2)->array();
      classifier.get_node_bias(1)->array() -= eta * \
      classifier.get_node_delta(1)->array();
    }
    //classifier.Evaluate(test_images_ptr, test_labels_ptr);
    classifier.set_count(0);
  }
  std::cout << "Complete" << std::endl;
  */
  return 0;
}
