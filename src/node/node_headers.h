/* Copyright 2019 The Nicole Authors. All Rights Reserved.
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
// Contains node headers

#include "node/identity_l2_node.h"
#include "node/identity_node.h"
#include "node/node_factory.h"
#include "node/node_parameter.h"
#include "node/node_registry.h"
#include "node/node.h"
#include "node/output_node.h"
#include "node/relu_node.h"
#include "node/sigmoid_cross_entropy.h"
#include "node/sigmoid_l2_node.h"
#include "node/sigmoid_node.h"
#include "node/softmax_log_node.h"
#include "node/tanh_node.h"