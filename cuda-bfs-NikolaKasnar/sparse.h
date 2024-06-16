#pragma once 
#include "labirint_io.h"
// NE TREBA NIŠTA MIJENJATI.

// Rijetka matrica komprimirana po recima. 
// Matrica u CSR formatu (compressed sparse row)
struct CSRMat{
    // Konstruiraj CSR matricu iz pune matrice.
	CSRMat(IncidenceMat const &);
	~CSRMat(); 
	void print() const;
	
    int * rowPtrs;
	int * colIdx;

	int nelem = 0;
	int nrows = 0;
	int ncols = 0;
};


// Rijetka matrica komprimirana po stupcima. 
// Matrica u CSC formatu (compressed sparse column)
struct CSCMat{
    // Konstruiraj CSC matricu iz pune matrice.
	CSCMat(IncidenceMat const &);
	~CSCMat(); 
	void print() const;
	
    int * colPtrs;
	int * rowIdx;

	int nelem = 0;
	int nrows = 0;
	int ncols = 0;
};




