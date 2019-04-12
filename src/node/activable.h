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
#ifndef INTELLGRPAH_NODE_ACTIVABLE_H_
#define INTELLGRPAH_NODE_ACTIVABLE_H_

#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {

// Activation matrix has three states and are used to determine behaviors of
// node methods.
enum ActStates {
  // Activation maxtix is initialized or overwritten with weighted_sum
  kInit = 0,
  // CallActFxn() is invoked and activation matrix is overwritten with
  // activation values.
  kAct = 1,
  // Dropout is applied on the activation matrix.
  kDropout = 2,
  // CalcPrime() is invoked and activation matrix is overwritten with prime
  // values.
  kPrime = 3,
  // Indicates current node is an input node.
  kFeed = 4,
};

template <class T>
interface activable {
 public:
  virtual ~activable() noexcept = default;

  virtual bool CallActFxn() = 0;

  virtual bool CalcActPrime() = 0;

  virtual void Evaluate(REF const Eigen::Ref<const MatXX<T>>& labels) = 0;

  virtual inline bool ResetActState() = 0;

  virtual inline void TurnDropoutOn(T dropout_p) = 0;

  virtual inline void TurnDropoutOff() = 0;
 
 protected:
  // Transitions from kInit to kAct and updates current_act_state_
  virtual void InitToAct() = 0;

  // Transitions from kAct state to kDropout state and updates current_act_state_
  virtual void ActToDropout() = 0;

  // Transitions from kDropout state to kPrime and updates current_act_state_
  virtual void DropoutToPrime() = 0;
  
  // Transitions from current_act_state_ to state
  virtual bool Transition(ActStates state) = 0;

};

}  // intellgraph

#endif  // INTELLGRPAH_NODE_ACTIVABLE_H_