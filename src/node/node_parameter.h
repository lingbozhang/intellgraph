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
#ifndef INTELLGRAPH_LAYER_NODE_PARAMETER_H_
#define INTELLGRAPH_LAYER_NODE_PARAMETER_H_

#include <string>
#include <utility>
#include <vector>

namespace intellgraph {
// NodeParameter contains node information and is used to build node object.
// Note NodeParameter is a movable class and it must follow rules for movable 
// type. 
// Specifically, in NodeParameter class, constructor and operator= only 
// accept rvalues. There are two versions of accessors:
//   * get_variable_name returns a copy of the variable
//   * get_k_variable_name returns a constant ref of the variable
// NodeParameter provides a Clone method which is used to copy from other object
// In NodeParameter, in order to implement method chaining, mutators return
// reference of corresponding object.
class NodeParameter {
 public:
  NodeParameter() {}

  explicit NodeParameter(size_t id, const std::string& name, \
                         const std::vector<size_t>& dims)
      : id_(id), node_name_(name), dims_(dims) {}

  // Default constructor is equivalent to member-wise move constructor
  NodeParameter(NodeParameter&& rhs) noexcept = default;

  NodeParameter& operator=(NodeParameter&& rhs) noexcept = default;
  
  void Clone(const NodeParameter& rhs) {
    id_ = rhs.get_k_id();
    dims_ = rhs.get_k_dims();
    node_name_ = rhs.get_k_node_name();
  }

  // Copy operations are explicitly deleted
  NodeParameter(const NodeParameter& rhs) = delete;
  NodeParameter& operator=(const NodeParameter& rhs) = delete;
  
  ~NodeParameter() = default;

  // Accessor function name with letter 'c' indicates return copy variables
  inline size_t get_c_id() const {
    return id_;
  }

  // Accessor function name with letter 'k' indicates return const variables refs
  inline const size_t get_k_id() const {
    return id_;
  }

  inline NodeParameter& set_c_id(size_t id) {
    id_ = id;
    return *this;
  }

  inline std::string get_c_node_name() const {
    return node_name_;
  }

  inline const std::string& get_k_node_name() const {
    return node_name_;
  }

  // Setters named with letter 'c' indicates a copy setter
  inline NodeParameter& set_c_node_name(const std::string& node_name) {
    node_name_ = node_name;
    return *this;
  }

  // Setters named with letter 'm' indicates a move setter
  inline NodeParameter& set_m_node_name(std::string&& node_name) {
    node_name_ = std::move(node_name);
    return *this;
  }

  inline std::vector<size_t> get_c_dims() const {
    return dims_;
  }

  inline const std::vector<size_t>& get_k_dims() const {
    return dims_;
  }

  inline NodeParameter& set_c_dims(const std::vector<size_t>& dims) {
    dims_ = dims;
    return *this;
  }

  // Setters named with letter 'm' indicates a move setter
  inline NodeParameter& set_m_dims(std::vector<size_t>&& dims) {
    dims_ = std::move(dims);
    return *this;
  }

 private:
  // List initialization (since C++11)
  // Note Member will be initialized before the class constructor
  size_t id_{0};
  std::string node_name_{""};
  std::vector<size_t> dims_{};
};

}  // namespace intellgraph

#endif  // INTELLGRAPH_LAYER_NODE_PARAMETER_H_