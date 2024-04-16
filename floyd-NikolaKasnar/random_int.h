#include <random>
#include <chrono>
#include <iostream>

// Klasa koja predstavlja generator slučajnih brojeva tipa int uniformo distribuiranih 
// u zadanom rasponu. 
template <typename T=int>
class RandomInt{
    using PT = std::uniform_int_distribution<T>::param_type;

    void seed_random(){
       // random_device koristimo samo za inicijalizaciju generatora
       // Danas je to uglavnom nedeterministički izvor.
         std::random_device  rd; //("/dev/urandom");
         re.seed(rd());
    } // Falila je ova zagrada
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

