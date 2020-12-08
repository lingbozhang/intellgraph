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
#include <iostream>

#include "examples/example1.h"
#include "glog/logging.h"

using namespace intellgraph;

int main(int argc, char *argv[]) {
  // Initializes Google's logging library.
  FLAGS_alsologtostderr = false;
  FLAGS_minloglevel = 0;
  fLS::FLAGS_log_dir = "/tmp/";
  google::InitGoogleLogging(argv[0]);

  Example1::Run();
  return 0;
}
