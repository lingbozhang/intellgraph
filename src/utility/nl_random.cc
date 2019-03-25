//
// Created by Lingbo Zhang on 3/11/19.
//
#include "nl_random.h"
#include <random>

double standard_normald(double dummy)
{
   static std::mt19937 rng;
   static std::normal_distribution<> nd(0, 1);
   return nd(rng);
}

float standard_normalf(float dummy)
{
   static std::mt19937 rng;
   static std::normal_distribution<> nd(0, 1);
   return nd(rng);
}

