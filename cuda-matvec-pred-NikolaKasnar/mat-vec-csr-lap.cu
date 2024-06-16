#include "csr_mat.h" 
#include "lapmat.h"

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

// Svaka nit radi na jednom retku matrice.
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
}

void error_h(cudaError_t error){
     if(error != cudaSuccess){
         std::cout << cudaGetErrorString(error) << "\n";
         std::exit(EXIT_FAILURE);
     }
}

int main(){
  const int N = 100;
  CSRMatrix<float> lap;
  LaplaceMatrix(N, lap);
  assert(lap.nRows == lap.nCols);

  if(lap.nRows < 100)
      printFull(lap);

  const int K = lap.nCols;  // dimenzija matrice
  std::cout << "Dimenzija matrice = " << K << "\n";
  float * x  = new float[K];
  for(unsigned int i=0; i<K; ++i) x[i]= 1;
  float * y  = new float[K];
  float * y1 = new float[K];

  // množenje na CPU
  spmv_csr(lap, x, y);

  // CUDA KOD
  CSRMatrix<float> * d_lap;
  // Alociram klasu na GPU i kopiram ju tamo. Pokazivači koje sam kopirao nemaju smislene vrijednosti.
  error_h( cudaMalloc((void **)(&d_lap), sizeof(CSRMatrix<float>)) );
  error_h( cudaMemcpy(d_lap, &lap, sizeof(CSRMatrix<float>), cudaMemcpyHostToDevice) );
  
  // Alociram na GPU polja koja se nalaze u klasi  CSRMatrix<float> s korektnim dimenzijama
  int * d_rowPtrs, *d_colIdx, *d_value; // Na device-u
  // Moram koristiti nove pokazivače jer ne mogu koristiti a CPU d_m_csr->rowPtrs itd,
  // jer su to pokazivači na GPU.
  error_h( cudaMalloc(&d_rowPtrs, (lap.nRows+1)*sizeof(int)) );
  error_h( cudaMalloc(&d_colIdx,   lap.nElem*sizeof(int)) );
  error_h( cudaMalloc(&d_value,    lap.nElem*sizeof(float)) );
  

  // Kopiram vrijednosti polja na GPU
  error_h( cudaMemcpy(d_rowPtrs, lap.rowPtrs, (lap.nRows+1)*sizeof(int),   cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(d_colIdx , lap.colIdx , (lap.nElem  )*sizeof(int),   cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(d_value  , lap.value  , (lap.nElem  )*sizeof(float), cudaMemcpyHostToDevice) );

  // Sada postavljam pokazivače unutar moje klase. 
  error_h( cudaMemcpy(&(d_lap->rowPtrs), &d_rowPtrs , sizeof(int*), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(&(d_lap->colIdx ), &d_colIdx  , sizeof(int*), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(&(d_lap->value),   &d_value , sizeof(float*), cudaMemcpyHostToDevice) );

  // Isto za vektore x i y1
  float *d_x, *d_y1;
  error_h( cudaMalloc(&d_x,  K*sizeof(float)) );
  error_h( cudaMalloc(&d_y1, K*sizeof(float)) );
  error_h( cudaMemcpy(d_x,   x, K*sizeof(float), cudaMemcpyHostToDevice) );
  error_h( cudaMemcpy(d_y1, y1, K*sizeof(float), cudaMemcpyHostToDevice) );

  //////////////////////////////////////////
  const int BLOCK = 128; 
  const int GRID = (K+BLOCK-1)/BLOCK;
  spmv_csr_kernel<<<GRID,BLOCK>>>(d_lap, d_x, d_y1);
  //////////////////////////////////////////

  error_h( cudaMemcpy(y1, d_y1, K*sizeof(float), cudaMemcpyDeviceToHost) );
  
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


  error_h( cudaFree(d_lap) ); error_h( cudaFree(d_x) ); error_h( cudaFree(d_y1) );
  delete [] x; delete [] y; delete [] y1;
  return 0;
}
