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
#include <vector>

namespace intellgraph {
// LayerParamter contains node information and is used to build node object
struct NodeParameter {
  size_t id;
  std::string fxn_name;
  std::vector<size_t> dims;
};

}  // namespace intellgraph

#endif  // INTELLGRAPH_LAYER_NODE_PARAMETER_H_