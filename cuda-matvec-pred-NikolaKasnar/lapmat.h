#pragma once
#include "csr_mat_base.h"
#include <cassert>

// Kreiraj matricu Laplaceovog operatora na uniformnoj mreži u
// tri dimenzije. Rubni uvjet je homogeni Dirichletov.
// N  = karakteristika mreže. Mreža ima   N+2 točke u svakom smjeru,
//      dimenzija matrice je N^3
//  M je kreirana matrica CSR formatu
template <typename T>
void LaplaceMatrix(unsigned int N, CSRMatrixBase<T> & M)
{ 
    M.resize(N*N*N, N*N*N, 7*N*N*N - 2*(N*N+N+1));
    assert(M.rowPtrs);
    assert(M.colIdx);
    assert(M.value);

    // Postavi pokazivače na početak reda
	M.rowPtrs[0] = 0;
	M.rowPtrs[1] = 4;
    for(unsigned int i=2; i <= N; ++i)
        M.rowPtrs[i] = M.rowPtrs[i-1] + 5;

    for(unsigned int i=N+1; i <= N*N; ++i)
        M.rowPtrs[i] = M.rowPtrs[i-1] + 6;
	 
    for(unsigned int i=N*N+1; i <= N*N*N - N*N; ++i)
        M.rowPtrs[i] = M.rowPtrs[i-1] + 7;

    for(unsigned int i=N*N*N - N*N+1; i <= N*N*N - N; ++i)
        M.rowPtrs[i] = M.rowPtrs[i-1] + 6;
	
    for(unsigned int i=N*N*N - N+1; i <= N*N*N - 1; ++i)
        M.rowPtrs[i] = M.rowPtrs[i-1] + 5;
    // Kraj zadnjeg reda
    M.rowPtrs[N*N*N] = M.rowPtrs[N*N*N-1] + 4;  //  = M.nElem

    assert(M.rowPtrs[N*N*N] == M.nElem);

	for(unsigned int row=0; row< M.nRows; ++row){
		auto beg = M.rowPtrs[row];
		auto end = M.rowPtrs[row+1];
		M.colIdx[beg] = row;  // Prvi element u svakom redu je dijagonalan
		M.value[beg] = 6.0;
		for(int j=beg+1; j<end; ++j)
			M.value[j] = -1.0;
	}
    // Prvi red
    M.colIdx[1] = 1; M.colIdx[2] = N; M.colIdx[3] = N*N;
    for(unsigned int row=1; row < N; ++row){
		auto beg = M.rowPtrs[row];
		auto end = M.rowPtrs[row+1];
		M.colIdx[beg+1] = row-1; 
		M.colIdx[beg+2] = row+1; 
		M.colIdx[beg+3] = row+N;
		M.colIdx[beg+4] = row+N*N;
	}

    for(unsigned int row=N; row < N*N; ++row){
		auto beg = M.rowPtrs[row];
		auto end = M.rowPtrs[row+1];
		M.colIdx[beg+1] = row-N; 
		M.colIdx[beg+2] = row-1; 
		M.colIdx[beg+3] = row+1; 
		M.colIdx[beg+4] = row+N;
		M.colIdx[beg+5] = row+N*N;
	}
	for(unsigned int row=N*N; row < N*N*N - N*N; ++row){
		auto beg = M.rowPtrs[row];
		auto end = M.rowPtrs[row+1];
		M.colIdx[beg+1] = row-N*N; 
		M.colIdx[beg+2] = row-N; 
		M.colIdx[beg+3] = row-1; 
		M.colIdx[beg+4] = row+1;
		M.colIdx[beg+5] = row+N;
		M.colIdx[beg+6] = row+N*N;
	}
	for(unsigned int row=N*N*N - N*N; row < N*N*N - N; ++row){
		auto beg = M.rowPtrs[row];
		auto end = M.rowPtrs[row+1];
		M.colIdx[beg+1] = row-N*N; 
		M.colIdx[beg+2] = row-N; 
		M.colIdx[beg+3] = row-1; 
		M.colIdx[beg+4] = row+1;
		M.colIdx[beg+5] = row+N;
	}
	for(unsigned int row=N*N*N - N; row < N*N*N - 1; ++row){
		auto beg = M.rowPtrs[row];
		auto end = M.rowPtrs[row+1];
		M.colIdx[beg+1] = row-N*N; 
		M.colIdx[beg+2] = row-N; 
		M.colIdx[beg+3] = row-1; 
		M.colIdx[beg+4] = row+1;
	}
	// Zadnji red
	auto row = N*N*N-1;
	auto beg = M.rowPtrs[row]; 
	M.colIdx[beg+1] = row-N*N; 
	M.colIdx[beg+2] = row-N; 
	M.colIdx[beg+3] = row-1;
}
