#ifndef NICOLE_LAYER_TEST_BASE_LBZ_H_
#define NICOLE_LAYER_TEST_BASE_LBZ_H_

template<class T, typename Implementation>
class Base_lbz {
 public:
  Base_lbz(){};
  ~Base_lbz(){};

  void Print_lbz() {
    Impl().Print();
  }
 
 private:
  Implementation& Impl() {
    return *static_cast<Implementation*>(this);
  }
};

#endif