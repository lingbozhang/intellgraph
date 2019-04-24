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
#ifndef INTELLGRAPH_NODE_NODE_PARAMETER_H_
#define INTELLGRAPH_NODE_NODE_PARAMETER_H_

#include <string>
#include <vector>

#include "utility/auxiliary_cpp.h"

namespace intellgraph {

// NodeParameter contains node information and is used to build node object.
// Note NodeParameter is a movable class and it must follow rules for movable 
// type. 
// Specifically, in NodeParameter class, constructor and operator= only 
// accept rvalues. There are three versions of accessors:
//   * get_variable_name returns a copy of the variable
//   * ref_variable_name returns a constant reference of the variable
// There are two versions of mutators:
//   * set_variable_name sets a variable by copy
//   * move_variable_name sets a variable by move
// NodeParameter provides a Clone method which is used to copy from other object
// In NodeParameter, in order to implement method chaining, mutators return
// reference of corresponding object.
class NodeParameter {
 public:
  NodeParameter() noexcept {};

  explicit NodeParameter(COPY size_t id, REF const std::string& name, \
                         REF const std::vector<size_t>& dims, \
                         REF const std::string& activator_name = "", \
                         REF const std::string& estimator_name = "", \
                         REF const std::vector<size_t>& node_struct = {})
      : id_(id), node_name_(name), dims_(dims), \
        activator_name_(activator_name), estimator_name_(estimator_name), \
        node_struct_(node_struct) {}

  // Default constructor is equivalent to member-wise move constructor
  NodeParameter(MOVE NodeParameter&& rhs) noexcept = default;

  REF NodeParameter& operator=(MOVE NodeParameter&& rhs) noexcept = default;
  
  // Copy operations are explicitly deleted
  NodeParameter(REF const NodeParameter& rhs) = delete;
  REF NodeParameter& operator=(REF const NodeParameter& rhs) = delete;

  inline void Clone(REF const NodeParameter& rhs) {
    id_ = rhs.ref_id();
    dims_ = rhs.ref_dims();
    node_name_ = rhs.ref_node_name();
    activator_name_ = rhs.ref_activator_name();
    estimator_name_ = rhs.ref_estimator_name();
    node_struct_ = rhs.ref_node_struct();
  }

  ~NodeParameter() noexcept = default;

  COPY inline size_t get_id() const {
    return id_;
  }

  REF inline const size_t ref_id() const {
    return id_;
  }

  REF inline NodeParameter& set_id(COPY size_t id) {
    id_ = id;
    return *this;
  }

  COPY inline std::string get_node_name() const {
    return node_name_;
  }

  REF inline const std::string& ref_node_name() const {
    return node_name_;
  }

  inline NodeParameter& set_node_name(REF const std::string& node_name) {
    node_name_ = node_name;
    return *this;
  }

  inline NodeParameter& move_node_name(MOVE std::string&& node_name) {
    node_name_ = std::move(node_name);
    return *this;
  }

  COPY inline std::vector<size_t> get_dims() const {
    return dims_;
  }

  REF inline const std::vector<size_t>& ref_dims() const {
    return dims_;
  }

  inline NodeParameter& set_dims(REF const std::vector<size_t>& dims) {
    dims_ = dims;
    return *this;
  }

  inline NodeParameter& move_dims(MOVE std::vector<size_t>&& dims) {
    dims_ = std::move(dims);
    return *this;
  }
  
  COPY inline std::string get_activator_name() const {
    return activator_name_;
  }

  REF inline const std::string& ref_activator_name() const {
    return activator_name_;
  }

  inline NodeParameter& set_activator_name(REF const std::string& name) {
    activator_name_ = name;
    return *this;
  }

  inline NodeParameter& move_activator_name(MOVE std::string&& name) {
    activator_name_ = std::move(name);
    return *this;
  }

  COPY inline std::string get_estimator_name() const {
    return estimator_name_;
  }

  REF inline const std::string& ref_estimator_name() const {
    return estimator_name_;
  }

  inline NodeParameter& set_estimator_name(REF const std::string& name) {
    estimator_name_ = name;
    return *this;
  }

  inline NodeParameter& move_estimator_name(MOVE std::string&& name) {
    estimator_name_ = std::move(name);
    return *this;
  }

  REF inline const std::vector<size_t>& ref_node_struct() const {
    return node_struct_;
  }

  inline NodeParameter& set_node_struct(REF const std::vector<size_t>& node_struct) {
    node_struct_ = node_struct;
    return *this;
  }

  inline NodeParameter& move_node_struct(MOVE std::vector<size_t>&& node_struct) {
    node_struct_ = std::move(node_struct);
    return *this;
  }

 private:
  // List initialization (since C++11)
  // Note Member will be initialized before the class constructor (C++11)
  size_t id_{0};
  std::string node_name_{""};
  std::vector<size_t> dims_{};

  std::string activator_name_{""};
  std::string estimator_name_{""};

  std::vector<size_t> node_struct_{};

};

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_NODE_PARAMETER_H_