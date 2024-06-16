#pragma once
#include <cassert>
#include "csr_mat_base.h"
#include <cuda_runtime.h>

// Zapis matrice u rijetkom retčanom formatu, CSR.
// "Compressed sparse row storage" (CSR).
// Verzija s optimiziranom unificiranom memorijom.
template <typename T>
struct CSRMatrixUM : public CSRMatrixBase<T>{
    // prazna matrica
    CSRMatrixUM() = default; 
	~CSRMatrixUM(){ clear(); }
	void clear();
    // alociraj matricu nove dimenzije
    void resize(int nrows_, int ncols_, int nelem_) override;
};

template <typename T>
void CSRMatrixUM<T>::resize(int nrows_, int ncols_, int nelem_)
{
    clear();
    this->nElem = nelem_;
    this->nRows = nrows_;
    this->nCols = ncols_;
	cudaMallocManaged(&this->rowPtrs, (nrows_+1)*sizeof(int));
	cudaMallocManaged(&this->colIdx,   nelem_*sizeof(int));
	cudaMallocManaged(&this->value,    nelem_*sizeof(T));

    cudaMemPrefetchAsync(this->rowPtrs, (nrows_+1)*sizeof(int), cudaCpuDeviceId);
    cudaMemPrefetchAsync(this->colIdx,  nelem_*sizeof(int), cudaCpuDeviceId);
    cudaMemPrefetchAsync(this->value,   nelem_*sizeof(T), cudaCpuDeviceId);
}


// Dealociraj memoriju.
template <typename T>
void CSRMatrixUM<T>::clear(){
	cudaFree(this->value);
	cudaFree(this->rowPtrs);
	cudaFree(this->colIdx);
	this->value = nullptr;
	this->rowPtrs = nullptr;
	this->colIdx = nullptr;
	this->nElem = 0;
	this->nRows = 0;
	this->nCols = 0;
}

