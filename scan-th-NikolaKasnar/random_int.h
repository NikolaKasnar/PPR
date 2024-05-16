#pragma once

#include <random>
#include <chrono>
#include <iostream>

// Klasa koja predstavlja generator slučajnih brojeva tipa int uniformo distribuiranih 
// u zadanom rasponu.  Ne dirati. 
template <typename T=int>
class RandomInt{
    using PT = std::uniform_int_distribution<T>::param_type;

    void seed_random(){
         std::random_device  rd; 
         re.seed(rd());
    }
public:
   RandomInt(T a, T b) {
       seed_random();
       d.param(PT{a,b});
   }
   RandomInt(RandomInt const & rhs) : re(rhs.re), d(rhs.d){
       seed_random();
   }
   RandomInt & operator=(RandomInt const & rhs) = delete;

   T operator()(){ return d(re); }
private:
    std::mt19937 re; 
    std::uniform_int_distribution<T> d;
};

