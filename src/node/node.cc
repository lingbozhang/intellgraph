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
#include "node/node.h"

namespace intellgraph {

template <class T>
Node<T>::Node(const NodeParameter& node_param) {
  node_param_.Clone(node_param);

  size_t row = node_param.ref_dims()[0];

  bias_ptr_ = std::make_unique<VecX<T>>(row);

  bias_ptr_->array() = 0.0;

  state_ptr_ = std::move(InitialState<T>::get_instance());
}

template class Node<float>;
template class Node<double>;

template <class T>
void ActState<T>::ChangeState(Node<T>* node, ActStateUPtr<T> state_ptr) {
  LOG(INFO) << "Node: " << node->ref_node_id()
            << ", Transitions to " << state_ptr->get_state_name();
  node->ChangeState(std::move(state_ptr));
}

template class ActState<float>;
template class ActState<double>;

template <class T>
void InitialState<T>::ToInit(Node<T>* node) {
  this->ChangeState(node, InitialState::get_instance());
}

template <class T>
void InitialState<T>::ToFeed(Node<T>* node) {
  this->ChangeState(node, FeedState<T>::get_instance());
}

template <class T>
void InitialState<T>::ToAct(Node<T>* node) {
  LOG(INFO) << "Node: " << node->ref_node_id()
            << ", Activation is activated";
  node->Activate();
  this->ChangeState(node, ActivateState<T>::get_instance());
}

template class InitialState<float>;
template class InitialState<double>;

template <class T>
void FeedState<T>::ToInit(Node<T>* node) {
  this->ChangeState(node, InitialState<T>::get_instance());
}

template <class T>
void FeedState<T>::ToAct(Node<T>* node) {
  LOG(INFO) << "Node: " << node->ref_node_id()
            << ", ToAct() is called, but state remains in FeedState";
}

template <class T>
void FeedState<T>::ToPrime(Node<T>* node) {
  LOG(INFO) << "Node: " << node->ref_node_id()
            << ", ToPrime() is called, but state remains in FeedState";
}

template <class T>
void FeedState<T>::ToFeed(Node<T>* node) {
  LOG(INFO) << "Node: " << node->ref_node_id()
            << ", ToFeed() is called, but state remains in FeedState";
}

template class FeedState<float>;
template class FeedState<double>;

template <class T>
void State<T>::ToInit(Node<T>* node) {
  this->ChangeState(node, InitialState<T>::get_instance());
}

template <class T>
void ActivateState<T>::ToPrime(Node<T>* node) {
  if(!node->ref_dropout_on()) {
    LOG(INFO) << "Node: " << node->ref_node_id()
              << ", Prime is calculated";
    node->Prime();
    this->ChangeState(node, PrimeState<T>::get_instance());
  } else {
    LOG(ERROR) << "ToPrime() for ActivateState is failed";
    exit(1);
  }
}

template <class T>
void ActivateState<T>::ToDropout(Node<T>* node) {
  LOG(INFO) << "Node: " << node->ref_node_id()
            << ", Dropout is applied";
  node->Dropout();
  this->ChangeState(node, DropoutState<T>::get_instance());
}

template class ActivateState<float>;
template class ActivateState<double>;

template <class T>
void DropoutState<T>::ToPrime(Node<T>* node) {
  LOG(INFO) << "Node: " << node->ref_node_id()
            << ", Prime is calculated";
  node->Prime();
  this->ChangeState(node, PrimeState<T>::get_instance());
}

template class DropoutState<float>;
template class DropoutState<double>;

template <class T>
void PrimeState<T>::ToInit(Node<T>* node) {
  this->ChangeState(node, InitialState<T>::get_instance());
}

template class PrimeState<float>;
template class PrimeState<double>;

}  // intellgraph