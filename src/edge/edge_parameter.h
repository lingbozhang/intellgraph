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
#ifndef INTELLGRAPH_EDGE_EDGE_PARAMETER_H_
#define INTELLGRAPH_EDGE_EDGE_PARAMETER_H_

#include <string>
#include <vector>

namespace intellgraph {
// EdgeParamter contains edge information and is used to construct the edge 
// object
class EdgeParameter {
 public:
  EdgeParameter() noexcept {};

  explicit EdgeParameter(size_t id, const std::string& name, \
                         const std::vector<size_t>& dims_in, \
                         const std::vector<size_t>& dims_out)
      : id_(id), edge_name_(name), dims_in_(dims_in), dims_out_(dims_out) {}

  // Move constructor
  EdgeParameter(EdgeParameter&& rhs) noexcept = default;

  // Move operator
  EdgeParameter& operator=(EdgeParameter&& rhs) noexcept = default;

  // Copy operations are explicitly deleted
  EdgeParameter(const EdgeParameter& rhs) = delete;
  EdgeParameter& operator=(const EdgeParameter& rhs) = delete;

  inline void Clone(const EdgeParameter& rhs) {
    id_ = rhs.get_k_id();
    edge_name_ = rhs.get_k_edge_name();
    dims_in_ = rhs.get_k_dims_in();
    dims_out_ = rhs.get_k_dims_out();
  }

  ~EdgeParameter() noexcept = default;

  inline size_t get_c_id() const {
    return id_;
  }
 
  inline const size_t get_k_id() const {
    return id_;
  }

  inline EdgeParameter& set_c_id(size_t id) {
    id_ = id;
    return *this;
  }

  inline std::string get_c_edge_name() const {
    return edge_name_;
  }
 
  inline const std::string& get_k_edge_name() const {
    return edge_name_;
  }

  inline EdgeParameter& set_c_edge_name(const std::string& edge_name) {
    edge_name_ = edge_name;
    return *this;
  }

  inline EdgeParameter& set_m_edge_name(std::string&& edge_name) {
    edge_name_ = std::move(edge_name);
    return *this;
  }
  
  inline std::vector<size_t> get_c_dims_in() const {
    return dims_in_;
  }

  inline const std::vector<size_t>& get_k_dims_in() const {
    return dims_in_;
  }

  inline EdgeParameter& set_c_dims_in(const std::vector<size_t>& dims_in) {
    dims_in_ = dims_in;
    return *this;
  }

  inline EdgeParameter& set_m_dims_in(std::vector<size_t>&& dims_in) {
    dims_in_ = std::move(dims_in);
    return *this;
  }

  inline std::vector<size_t> get_c_dims_out() const {
    return dims_out_;
  }

  inline const std::vector<size_t>& get_k_dims_out() const {
    return dims_out_;
  }

  inline EdgeParameter& set_c_dims_out(const std::vector<size_t>& dims_out) {
    dims_out_ = dims_out;
    return *this;
  }

  inline EdgeParameter& set_m_dims_out(std::vector<size_t>&& dims_out) {
    dims_out_ = std::move(dims_out);
    return *this;
  }

 private:
  size_t id_{0};
  std::string edge_name_{""};
  std::vector<size_t> dims_in_{};
  std::vector<size_t> dims_out_{};
  
};

}  // namespace intellgraph
#endif  // INTELLGRAPH_EDGE_EDGE_PARAMETER_H_