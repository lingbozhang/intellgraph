#ifndef NICOLE_LAYER_TEST_DERIVED_H_
#define NICOLE_LAYER_TEST_DERIVED_H_
#include "layer/test_base.h"


template<class T>
class Derived : public Base_lbz<T, Derived<T>> {
 public:
  Derived(){}
  ~Derived(){}

  void Print();
 private:
};
#endif