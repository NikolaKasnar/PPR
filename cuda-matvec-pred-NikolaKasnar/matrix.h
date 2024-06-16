#pragma once

#include <cstdlib>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cassert>

// Puna matrica s elementima tipa T. Ovdje je minimalna implementacija.
// Trebam samo metode cols() i rows() te dohvat elemenata. Dodajemo i 
// funkciju print() za ispis malih matrica.
template <typename T>
class Matrix{
  public:
    Matrix(std::size_t rows, std::size_t cols);
    ~Matrix(){ if(data) delete [] data; }
    std::size_t rows() const { return nRows; }
    std::size_t cols() const { return nCols; }
    T operator()(int i, int j) const { return data[i*nCols+j]; }
    T& operator()(int i, int j) { return data[i*nCols+j]; }
  private:
    T * data;
    std::size_t nRows;
    std::size_t nCols;
};

template <typename T>
Matrix<T>::Matrix(std::size_t i, std::size_t j) : nRows(i), nCols(j) {
    assert(nRows >0);
    assert(nCols >0);

    data = new T[nRows*nCols];
    if(!data)
      throw std::runtime_error("Allocation error. Size = " + std::to_string(nRows*nCols));

    for(std::size_t i=0; i<nRows*nCols; ++i)
      data[i] = 0;
}

template <typename T>
void print(Matrix<T> const & M, std::ostream & out = std::cout, int width=2){
  for(int i=0; i< M.rows(); ++i){
    for(int j=0; j<M.cols(); ++j){
        out.width(width);
        out << M(i,j) << " ";
    }
    out << "\n";
  }
}
