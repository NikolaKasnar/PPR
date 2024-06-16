#include "csr_mat_um_pre.h" 
#include "lapmat.h"
#include <iostream>

__host__ 
void spmv_csr(CSRMatrixUM<float> const * csrMat, const float * x, float * y){
    for(int row = 0; row < csrMat->nRows; ++row){
        float sum = 0.0f;
        for(int i = csrMat->rowPtrs[row]; i < csrMat->rowPtrs[row+1]; ++i){
             int col = csrMat->colIdx[i];
             float value = csrMat->value[i];
             sum += value*x[col];
        }
        y[row] = sum;
    }

}

// Svaka nit radi na jednom retku matrice.
__global__ 
void spmv_csr_kernel(const CSRMatrixUM<float> csrMat, float *x, float * y){
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int nRows = csrMat.nRows;
    int * rowPtrs = csrMat.rowPtrs;
    int * colIdx  = csrMat.colIdx;
    float * val   = csrMat.value;
    if(row < nRows){
        float sum = 0.0f;
        //printf("rowPtrs[row] = %d, rowPtrs[row+1] = %d\n",rowPtrs[row], rowPtrs[row+1] );
        for(int i = rowPtrs[row]; i < rowPtrs[row+1]; ++i){
             int col = colIdx[i];
             float value = val[i];
             sum += value*x[col];
        }
        y[row] = sum; 
    }
    //printf("row = %d, nRows= %d\n", row, nRows);
}

void error_h(cudaError_t error){
     if(error != cudaSuccess){
         std::cout << cudaGetErrorString(error) << "\n";
         std::exit(EXIT_FAILURE);
     }
}

int main()
{
    const int N = 100;
    CSRMatrixUM<float> lap;
    LaplaceMatrix(N, lap);
    assert(lap.nRows == lap.nCols);
    const int K = lap.nCols;  // dimenzija matrice 
    float * x, * y1;
    cudaMallocManaged(&x, K*sizeof(float));
    cudaMallocManaged(&y1, K*sizeof(float));
    cudaMemAdvise(x, K*sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    int id;
    cudaGetDevice(&id);
    cudaMemPrefetchAsync(y1, K*sizeof(float), id);

    cudaMemAdvise(lap.rowPtrs, (lap.nRows+1)*sizeof(int), cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(lap.colIdx,   lap.nElem*sizeof(int),    cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(lap.value,    lap.nElem*sizeof(float),  cudaMemAdviseSetReadMostly, id);

    cudaMemPrefetchAsync(lap.rowPtrs, (lap.nRows+1)*sizeof(int), id);
    cudaMemPrefetchAsync(lap.colIdx,   lap.nElem*sizeof(int), id);
    cudaMemPrefetchAsync(lap.value,    lap.nElem*sizeof(float), id);

    for(unsigned int i=0; i<K; ++i) x[i]= 1;
    
    cudaMemAdvise(x, K*sizeof(float), cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(x, K*sizeof(float), id);

    float * y  = new float[K]; 
    // množenje na CPU
    spmv_csr(&lap, x, y);

    // CUDA KOD
    //////////////////////////////////////////
    const int BLOCK = 128; 
    const int GRID = (K+BLOCK-1)/BLOCK;
    spmv_csr_kernel<<<GRID,BLOCK>>>(lap, x, y1);
    cudaDeviceSynchronize();
    //////////////////////////////////////////
 
    cudaMemPrefetchAsync(y1, K*sizeof(float), cudaCpuDeviceId);

    bool res = true;
    for(int i=0; i<K; ++i){
        if(std::abs(y[i] - y1[i]) > 1E-7){
            res = false;
            std::cerr << "Razlika u serijskom i pearalelnom množenju: serijski = "
                      << y[i] << ", paralelni = " << y1[i]<< " (i= " << i << ")\n";
            break; 
        }
    }
    std::cout << "Rezultat je ";
    if(res) std::cout << "točan.\n";
    else    std::cout << "pogrešan.\n";

//    std::cout << y1[0] << " " << y1[K/2]<< " "  << y1[K-1] << "\n";

    error_h( cudaFree(x) ); error_h( cudaFree(y1) );
    delete [] y;
    return 0;
}
