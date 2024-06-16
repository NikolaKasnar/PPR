#include "csr_mat.h"
#include "matrix.h"

__host__ 
void spmv_csr(CSRMatrix<float> const & csrMat, const float * x, float * y){
    for(int row = 0; row < csrMat.nRows; ++row){
        float sum = 0.0f;
        for(int i = csrMat.rowPtrs[row]; i < csrMat.rowPtrs[row+1]; ++i){
             int col = csrMat.colIdx[i];
             float value = csrMat.value[i];
             sum += value*x[col];
        }
        y[row] = sum;
    }

}

__global__ 
void spmv_csr_kernel(const CSRMatrix<float> * csrMat, float *x, float * y){
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int nRows = csrMat->nRows;
    int * rowPtrs = csrMat->rowPtrs;
    int * colIdx  = csrMat->colIdx;
    float * val   = csrMat->value;
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

int main(){
  const int N = 5;
  Matrix<float> m(N,N);
  m(0,0) = 1; m(0,2) = 3;
  m(1,1) = 2; m(1,3) = 3; m(1,4) = 1;
  m(2,2) = 1;
  m(3,1) = 2; m(3,3) = 1; m(3,4) = 1;
  m(4,3) = 1;
  std::cout << "Puna matrica:\n";
  print(m);

  float * x  = new float[N]{1,1,1,1,1};
  float * y  = new float[N];
  float * y1 = new float[N];
  
  CSRMatrix<float> m_csr;
  convertToCSR(m, m_csr);
  std::cout << "CSR matrica:\n";
  print(m_csr);

  spmv_csr(m_csr, x, y);

  std::cout << "Vektor Mx:\n";
  for(int i=0; i<5; ++i)
     std::cout << y[i]  << "\n";
  
  // CUDA KOD
  CSRMatrix<float> * d_m_csr;
  float *d_x, *d_y1;
  // Alociram klasu na GPU i kopiram ju tamo. Pokazivači koje sam kopirao nemaju smislene vrijednosti.
  error_h( cudaMalloc((void **)(&d_m_csr), sizeof(CSRMatrix<float>)) );
  error_h( cudaMemcpy(d_m_csr, &m_csr  , sizeof(CSRMatrix<float>), cudaMemcpyHostToDevice) );
  
  // Alociram na GPU polja koja se nalaze u klasi  CSRMatrix<float> s korektnim dimenzijama
  int * d_rowPtrs, *d_colIdx, *d_value; // Na device-u
  // Moram koristiti nove pokazivače jer ne mogu koristiti a CPU d_m_csr->rowPtrs itd,
  // jer su to pokazivači na GPU.
  error_h( cudaMalloc(&d_rowPtrs, (m_csr.nRows+1)*sizeof(int)) );
  error_h( cudaMalloc(&d_colIdx,      m_csr.nElem*sizeof(int)) );
  error_h( cudaMalloc(&d_value,       m_csr.nElem*sizeof(float)) );
  // Isto za vektore x i y1
  error_h( cudaMalloc(&d_x,  N*sizeof(float)) );
  error_h( cudaMalloc(&d_y1, N*sizeof(float)) );

  // Kopiram vrijednosti polja na GPU
  error_h( cudaMemcpy(d_rowPtrs, m_csr.rowPtrs, (m_csr.nRows+1)*sizeof(int), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(d_colIdx , m_csr.colIdx , (m_csr.nElem  )*sizeof(int), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(d_value  , m_csr.value  , (m_csr.nElem  )*sizeof(float), cudaMemcpyHostToDevice) );

  // Sada postavljam pokazivače unutar moje klase. 
  error_h( cudaMemcpy(&(d_m_csr->rowPtrs), &d_rowPtrs , sizeof(int*), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(&(d_m_csr->colIdx ), &d_colIdx  , sizeof(int*), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(&(d_m_csr->value), &d_value , sizeof(float*), cudaMemcpyHostToDevice) );

  error_h( cudaMemcpy(d_x,   x, N*sizeof(float), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(d_y1, y1, N*sizeof(float), cudaMemcpyHostToDevice) );

  //////////////////////////////////////////
  spmv_csr_kernel<<<1,32>>>(d_m_csr, d_x, d_y1);
  //////////////////////////////////////////

  error_h( cudaMemcpy(y1, d_y1, N*sizeof(float), cudaMemcpyDeviceToHost) );
  
  std::cout << "Vektor Mx paralelno:\n";
  for(int i=0; i<5; ++i)
     std::cout << y1[i]  << "\n";

  error_h( cudaFree(d_m_csr) ); error_h( cudaFree(d_x) ); error_h( cudaFree(d_y1) );
  delete [] x; delete [] y; delete [] y1;
  return 0;
}
