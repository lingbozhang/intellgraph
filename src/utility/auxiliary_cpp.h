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
#ifndef INTELLGRAPH_UTILITY_AUXILIARY_CPP_H_
#define INTELLGRAPH_UTILITY_AUXILIARY_CPP_H_

namespace intellgraph {
// This header file contains rules for IntellGraph developers.

// C++ language does not do very well in ownership like Rust. However, ownership
// is very important, it helps programmer keeps in mind of memory safety while
// programming. Therefore, in IntellGraph, several dummy specifiers are defined
// and developers are asked to use them in the header. Note it is usually not
// recommended to introduce macros in the header.

// Indicates pass/return by pointer, reference or shared_ptr
#define MUTE
// Indicates pass/return by copy
#define COPY
// Indicates pass/return by const or const reference/pointer
#define REF
// Indicates pass/return by rvalue, or unique_ptr in C++, move ownership is
// achieved with unique smart pointers.
#define MOVE

// In order to emphasize on the ownership exchange, in addition to the original
// accessor and mutator, new accessor and mutator are defined.
// ref_variable_name: returns variable reference
// move_variable_name: set variable by move

// In C++, abstract class is similar to interface in Java, in order to distinguish
// it with class, an interface macro is defined and should be used for abstract 
// class
#define interface class
// Corresponds to interface, the public is replaced with implements
#define implements public

// All interfaces should have virtual destructor in order to allow memory
// release from interfaces

}

#endif  // INTELLGRAPH_UTILITY_AUXILIARY_CPP_H_