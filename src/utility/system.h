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
#ifndef INTELLGRAPH_UTILITY_SYSTEM_H_
#define INTELLGRAPH_UTILITY_SYSTEM_H_

#include <string>
#include "utility/auxiliary_cpp.h"

// Get current working directory
COPY const std::string GetCWD() {
  // Notes path_lengh should not exceeds 255 character length
  char buffer[255];
  // Not sure if this works for Windows and Linux. Plans to add it in a kernel 
  // directory
  char *answer = getcwd(buffer, sizeof(buffer));
  std::string cwd;
  if (answer) {
    cwd = answer;
  }
  return cwd;
};

#endif  // INTELLGRAPH_UTILITY_SYSTEM_H_