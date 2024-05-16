#pragma once
// Pomoćne funkcije. Ne mijenjati.
#include <vector>
#include <cassert>
#include <iostream>

// Isprintaj kolekciju. 
template <typename T>
void print(std::vector<T> const & data){
    auto N = data.size();
    for(unsigned int n=0; n<N; ++n)
        std::cout << data[n] << ((n<N-1) ? "," : "\n");
}

// Usporedi dva vektora.
template <typename T>
bool compare(std::vector<T> const & data_1, std::vector<T> const & data_2){
     auto N_1 = data_1.size();
     auto N_2 = data_2.size();
     if(N_1 != N_2){
      std::cerr << "compare: size mismatch: " << N_1 << " != " << N_2 << "\n";
      return false;
     }
     for(unsigned i=0; i<N_1; ++i){
      if(data_1[i] != data_2[i]){
          std::cerr << "compare: mismatch at: " << i << "; " <<  data_1[i] << " != " << data_2[i] << "\n";
          return false;
      }
     }
     return true;
}

// Sekvencijalni scan algoritam.
// data = ulazni niz
// result = izlazni niz parcijalnih suma niza data.
template <typename T>
void seq_scan(std::vector<T> const & data, std::vector<T>  & result){
    auto N = data.size();
    assert(N>0);
    result.resize(N);
    result[0] = data[0];
    for(unsigned int n =1; n<N; ++n)
       result[n] = result[n-1] + data[n];
}

