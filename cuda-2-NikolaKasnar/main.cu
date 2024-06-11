#include <iostream>
#include <cuda.h>
#include <cmath>

#define R 2 // Definiramo radijus jezgre
#define TILE_WIDTH 4  // Output velicina (4x4)
#define BLOCK_SIZE (TILE_WIDTH + 2 * R)  // Input velicina (8x8)

// Definiramo vrijednost koju koristimo pri usporedbi matrica
// Oznacava dopustenu gresku jer rjesenja mogu odudarati radi zaokruzivanja
#define EPSILON 1e-6

// Definiramo konvolucijsku jezgru na konstantnoj memoriji
__constant__ float d_K[2*R+1][2*R+1];

// Neoptimizirana verzija konvolucijske jezgre
__global__
void convolutionKernel(float * A, float * B, int noRows, int noCols)
{
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0.0;

    for(int m=0; m<2*R+1; ++m){
        for(int n=0; n<2*R+1; ++n){
            int convRow = row + m - R;
            int convCol = col + n - R;
            if(convRow >= 0 && convRow < noRows && convCol >= 0 && convCol < noCols)
                result += d_K[m][n] * A[convRow * noCols + convCol];
        }
    }

    if(col < noCols && row < noRows)
         B[row*noCols+col] = result;
}

// Optimizirana jezgra koristeci dijeljenu memoriju
__global__
void convolutionKernelOptimized(float * A, float * B, int noRows, int noCols)
{
    // Definiramo dijeljenu memoriju za jedan "tile"
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - R;
    int col_i = col_o - R;

    // Prekopiramo "tile" iz globalne memorije na dijeljenju
    // Slican proces kao u prvoj dz sa cudom
    if (row_i >= 0 && row_i < noRows && col_i >= 0 && col_i < noCols) {
        shared_A[ty][tx] = A[row_i * noCols + col_i];
    } else {
        shared_A[ty][tx] = 0.0f; // Ako je rubni slucaj stavljamo 0
    }

    __syncthreads();

    float result = 0.0;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int m = 0; m < 2 * R + 1; ++m) {
            for (int n = 0; n < 2 * R + 1; ++n) {
                result += d_K[m][n] * shared_A[ty + m][tx + n];
            }
        }
        // Kopiramo rezultat na matricu B
        if (row_o < noRows && col_o < noCols) {
            B[row_o * noCols + col_o] = result;
        }
    }
}

// Funkcija za inicijaliziranje matrice
void initializeMatrices(float* A, float* B1, float* B2, int noRows, int noCols) {
    // Napunimo A sa nekim random vrijednostima, a B matrice sa nulama
    for(int i = 0; i < noRows * noCols; i++) {
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        B1[i] = 0.0f;
        B2[i] = 0.0f;
    }
}

// Funkcija za usporedbu dvije matrice
bool compareMatrices(float* B1, float* B2, int noRows, int noCols) {
    for (int i = 0; i < noRows * noCols; i++) {
        if (fabs(B1[i] - B2[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

int main(){
  // Dimenzije matrica
  int noRows = 1024;
  int noCols = 1024;

  // Alociramo memoriju na hostu
  float *h_A = new float[noRows * noCols];
  float *h_B_non_optimized = new float[noRows * noCols];
  float *h_B_optimized = new float[noRows * noCols];
  float h_K[5][5] = {
      0.1, 0.2, 0.3, 0.2, 0.1,
      0.2, 0.3, 0.4, 0.3, 0.2,
      0.3, 0.4, 0.5, 0.4, 0.3,
      0.2, 0.3, 0.4, 0.3, 0.2,
      0.1, 0.2, 0.3, 0.2, 0.1
  };

  // Inicijaliziramo matrice
  initializeMatrices(h_A, h_B_non_optimized, h_B_optimized, noRows, noCols);

  // Alociramo sve potrebno na GPU
  float *d_A, *d_B;
  cudaMalloc(&d_A, noRows * noCols * sizeof(float));
  cudaMalloc(&d_B, noRows * noCols * sizeof(float));

  cudaMemcpy(d_A, h_A, noRows * noCols * sizeof(float), cudaMemcpyHostToDevice);

  // Kopiramo konvolucijsku jezgru na konstantnu memoriju
  cudaMemcpyToSymbol(d_K, h_K, 5 * 5 * sizeof(float));

  // Definiramo block i grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((noCols + TILE_WIDTH - 1) / TILE_WIDTH, (noRows + TILE_WIDTH - 1) / TILE_WIDTH);

  // Varijable za mjerenje vremena
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Neoptimizirana verija
  cudaEventRecord(start);
  convolutionKernel<<<gridDim, blockDim>>>(d_A, d_B, noRows, noCols);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Neoptimizirana jezgra: " << milliseconds << " ms\n";

  // Kopiramo rezultat natrag na host
  cudaMemcpy(h_B_non_optimized, d_B, noRows * noCols * sizeof(float), cudaMemcpyDeviceToHost);

  // Optimizirana verzija
  cudaEventRecord(start);
  convolutionKernelOptimized<<<gridDim, blockDim>>>(d_A, d_B, noRows, noCols);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Optimizirana jezgra: " << milliseconds << " ms\n";

  cudaMemcpy(h_B_optimized, d_B, noRows * noCols * sizeof(float), cudaMemcpyDeviceToHost);

  if (compareMatrices(h_B_non_optimized, h_B_optimized, noRows, noCols)) {
    std::cout << "Matrice su iste.\n";
  } else {
    std::cout << "Matrice nisu iste.\n";
  }

  cudaFree(d_A);
  cudaFree(d_B);
  delete[] h_A;
  delete[] h_B_non_optimized;
  delete[] h_B_optimized;


  // Unistimo varijable za mjerenje vremena
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
