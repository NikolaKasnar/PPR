#include "sparse.h"
#include <iostream>
#include <cassert>
// DORADITI SAMO CSC KONSTRUKTOR.

// Konstruktor CSR matrice koji uzima punu matricu tipa IncidenceMat i konvertira ju u CSR matricu
CSRMat::CSRMat(IncidenceMat const & mat)
{
	  nrows = mat.rows();
	  ncols = mat.cols();
      assert(nrows == ncols);
      assert(nrows > 0);

	  rowPtrs = new int[nrows+1];
	  int cnt = 0;
      for(int i=0; i<mat.rows(); ++i)
		 for(int j=0; j<mat.cols(); ++j)
			 if(mat(i,j) != 0)
				 ++cnt;
	  nelem = cnt;
      assert(nelem >0);
	  colIdx = new int[cnt];
	  
	  cnt = 0;  // brojač ne nul elemenata
	  for(int i=0; i<nrows; ++i){
		  rowPtrs[i] = cnt;
		  for(int j=0; j<ncols; ++j){
             if(mat(i,j) != 0){
				 colIdx[cnt] = j;
				 ++cnt;
			 }           
		  }
	  }
      rowPtrs[nrows] = cnt;
	  assert(cnt == nelem);
	  std::cout << "CSRMat::CSRMat(Mat const & mat) nelem = " << nelem << "\n";
}

CSRMat::~CSRMat(){
	delete [] rowPtrs;
	delete [] colIdx; 
}

void CSRMat::print() const {
	std::cout << "rowPtrs = ";
	for(int i=0; i<=nrows; ++i)
		std::cout << rowPtrs[i]<<",";
	std::cout << "\ncolIdx = ";
	for(int i=0; i<nelem; ++i)
		std::cout << colIdx[i]<<",";
	std::cout << "\n";
}

// Konstruktor CSR matrice koji uzima punu matricu tipa IncidenceMat i konvertira ju u CSC matricu
CSCMat::CSCMat(IncidenceMat const & mat)
{
    // IMPLEMENTIRATI KONSTRUKTOR
    nrows = mat.rows();
    ncols = mat.cols();
    assert(nrows == ncols);
    assert(nrows > 0);

    // Prebrojimo broj ne-nul elemenata
    int cnt = 0;
    for(int j = 0; j < mat.cols(); ++j)
        for(int i = 0; i < mat.rows(); ++i)
            if(mat(i, j) != 0)
                ++cnt;
    nelem = cnt;
    assert(nelem > 0);

    // Alociramo memoriju za CSC format
    colPtrs = new int[ncols + 1];
    rowIdx = new int[nelem];

    // Popunimo nizove colPtrs i rowIdx
    cnt = 0;
    for(int j = 0; j < ncols; ++j) {
        colPtrs[j] = cnt;
        for(int i = 0; i < nrows; ++i) {
            if(mat(i, j) != 0) {
                rowIdx[cnt] = i;
                ++cnt;
            }
        }
    }
    colPtrs[ncols] = cnt;
    assert(cnt == nelem);
	std::cout << "CSCMat::CSCMat(Mat const & mat) nelem = " << nelem << "\n";
}

CSCMat::~CSCMat(){
	delete [] colPtrs;
	delete [] rowIdx; 
}

void CSCMat::print() const {
	std::cout << "colPtrs = ";
	for(int i=0; i<=nrows; ++i)
		std::cout << colPtrs[i]<<",";
	std::cout << "\nrowIdx = ";
	for(int i=0; i<nelem; ++i)
		std::cout << rowIdx[i]<<",";
	std::cout << "\n";
}


