#pragma once
#include "csr_mat_base.h"
#include <cassert>
#include <iostream>

// Zapis matrice u rijetkom retčanom formatu, CSR.
// "Compressed sparse row storage" (CSR).
template <typename T>
struct CSRMatrix : public CSRMatrixBase<T>{
    // prazna matrica
    CSRMatrix() = default; 
	~CSRMatrix(){ clear(); }
	void clear();
    // alociraj matricu nove dimenzije
    void resize(int nrows_, int ncols_, int nelem_) override;
};

template <typename T>
void CSRMatrix<T>::resize(int nrows_, int ncols_, int nelem_)
{
    clear();
    this->nElem = nelem_;
    this->nRows = nrows_;
    this->nCols = ncols_;
    this->rowPtrs = new int[nrows_+1]; 
    this->colIdx  = new int[nelem_];
	this->value   = new T[nelem_];
}


// Dealociraj memoriju.
template <typename T>
void CSRMatrix<T>::clear(){
	if(this->value)  
		delete [] this->value; 
	if(this->rowPtrs)
		delete [] this->rowPtrs; 
	if(this->colIdx)	
		delete [] this->colIdx; 
	this->value = nullptr;
	this->rowPtrs = nullptr;
	this->colIdx = nullptr;
	this->nElem = 0;
	this->nRows = 0;
	this->nCols = 0;
}

// Funkcija koja uzima punu matricu tipa Mat i konvertira ju u CSR matricu.
// Matrica tipa Mat mora imati metode rows(), cols() i operator (i,j) za
// dohvat elemenata.
// mat = ulazna puna matrica
// csrMat = izlaz, matrica mat zapisana u CSR formatu.
template <typename T, typename FullMat>
void convertToCSR(FullMat const & mat, CSRMatrix<T> & csrMat)
{
      int cnt = 0;
      for(int i=0; i<mat.rows(); ++i)
          for(int j=0; j<mat.cols(); ++j)
             if(mat(i,j) != 0)
                 ++cnt;

      csrMat.resize(mat.rows(), mat.cols(), cnt);

      cnt = 0;  // brojač ne-nul elemenata
      for(int i=0; i<csrMat.nRows; ++i){
          csrMat.rowPtrs[i] = cnt;
          for(int j=0; j<csrMat.nCols; ++j){
             if(mat(i,j) != 0){
                 csrMat.value[cnt] = mat(i,j);
                 csrMat.colIdx[cnt] = j;
                 ++cnt;
             }
          }
      }
      csrMat.rowPtrs[csrMat.nRows] = cnt;
      assert(cnt == csrMat.nElem);
      //	  std::cout << "CSRMat::CSRMat(Mat const & mat) nelem = " << nElem << "\n";
}
