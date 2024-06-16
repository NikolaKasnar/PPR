#pragma once 
#include <iostream>

// Zapis matrice u rijetkom retčanom formatu, CSR.
// "Compressed sparse row storage" (CSR).
// Apstraktna bazna klasa za različite verzije matrice.
template <typename T>
struct CSRMatrixBase{
    // alociraj matricu nove dimenzije
    virtual void resize(int nrows_, int ncols_, int nelem_) = 0;
    virtual ~CSRMatrixBase(){}

    int * rowPtrs = nullptr;
	  int *  colIdx = nullptr;
	  T   *   value = nullptr;

	  int nElem = 0;
	  int nRows = 0;
	  int nCols = 0;
};

// Ispiši CSR matricu u sirovom obliku.
template <typename T>
void print(CSRMatrixBase<T> const & M, std::ostream & out = std::cout, int width=2){
    out << "rowPtrs: ";
    for(int i=0; i<M.nRows+1; ++i)
          out << M.rowPtrs[i] << " ";
    out << "\ncolIdx = ";
    for(int i=0; i<M.nElem; ++i){
      out.width(width);
      out << M.colIdx[i] << " ";
    }
    out << "\nvalues = ";
    for(int i=0; i<M.nElem; ++i){
      out.width(width);
      out << M.value[i] << " ";
    }
    out << "\n";
}

// Ispiši CSR matricu kao punu matricu.
template <typename T>
void printFull(CSRMatrixBase<T> const & M, std::ostream & out = std::cout, int width=2){
    for(int row =0; row<M.nRows; ++row){
      for(int col=0; col<M.nCols; ++col){
             bool empty = true;
             for(int k=M.rowPtrs[row]; k<M.rowPtrs[row+1]; ++k){
                 if(col == M.colIdx[k]){
                     out.width(width);
                     out << M.value[k] << " ";
                     empty=false;
                 }
             }
             if(empty){
                 out.width(width);
                 out << 0 << " ";
             }
      }
      out << "\n";
    }
}