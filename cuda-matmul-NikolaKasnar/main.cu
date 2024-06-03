/*
   Kernel koji množi dvije kvadratne matrice. Matrice su dinamički alocirane i
   zadane su po recima. 
    C = A B
    Matrice su kvadratne dimenzije,  N x N. 
    Verzija s optimizacijom dohvata iz memorije (blok algoritam)
    i dimenzijom matrice koja ne mora biti djeljiva s dimenzijom bloka. 
*/

// Iostream je nedostajao za kasnije ispisvanje
#include <iostream>

// Definiramo block
#define BLOCK_SIZE 16

__global__
void matMulKernel(float * A, float * B, float * C, int N)
{
    // Matrice stavimo u dijeljenju memoriju
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Indexi za blokove
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Indexi za dretve
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Indexi reda i stupca sa kojima ova dretva radi
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Pocetna vrijednost elementa kojeg racunamo u C matrici
    float Cvalue = 0.0f;

    // Prolazimo po podmatricama
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Kopiramo elemente od A u dijeljenju memoriju As
        if (row < N && t * BLOCK_SIZE + tx < N)
            As[ty][tx] = A[row * N + t * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f; // U slucaju da je van granica punimo ju sa nulama

        // Kopiramo elemente od B u dijeljenju memoriju Bs
        if (col < N && t * BLOCK_SIZE + ty < N)
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f; // U slucaju da je van granica punimo ju sa nulama

        // Pricekamo da sve dretve zavrse za kopiranjem
        __syncthreads();

        // Mnozenje retka i stupca
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];

        // Pricekamo da sve dretve zavrse sa mnozenjem
        __syncthreads();
    }

    // Pisemo rezultat u globalnu memoriju
    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

void error_h(cudaError_t error, const char * file, int line){
     if(error != cudaSuccess){
         std::cerr << cudaGetErrorString(error) << " at file " << file << " in line " << line << "\n";
         std::exit(EXIT_FAILURE);
     }
}

__host__
bool checkResult(float * A, float * B, float * C, int N){
    float EPS = 1E-6f;
    bool res = true;
    float max_error = 0.0f;
    float max_value = 0.0f;
    for(int i=0; i<N; ++i)
    {
        for(int j=0; j<N; ++j)
        {
	       float tmp = 0.0;
           for(int k=0; k<N; ++k)
		       tmp += A[i*N+k] * B[k*N+j];

	       float diff = fabs(tmp - C[i*N+j]);
	       float val  = fabs(tmp);
	       if(diff > max_error) max_error = diff;
	       if(val  > max_value) max_value = val;  
	    } // for po j
    } // for po i

    
    if(max_error > EPS*max_value){
        std::cout << "(Max error = " << max_error << ", max value = " << max_value 
                  << ", relative error = " <<  max_error/max_value << ")";
        res = false;
    }
    return res;
}


int main()
{
    // Red matrice je N, u ovom slucaju nije djeljiv sa velicinom bloka koja je 16
    // Moze se testirati i sa ostalim velicinama
    int N = 1000;
    size_t bytes = N * N * sizeof(float);

    // Alociramo memoriju na hostu
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Popunimo matricu sa random brojevima koristeci funkciju rand()
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Alociramo memoriju na GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Kopiramo matrice sa h_A i b_A na d_A i d_B
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Postavimo blocks size i gridsize takod a radi i u slucaju kad N nije visekratnik od blocka
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Pozovem funkciju matMulKernel sa danim postavkama
    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Kopiram rezultat sa devicea na hosta
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Provjerim tocnost rezultata
    if (checkResult(h_A, h_B, h_C, N)) {
        std::cout << "Mnozenje matrica je tocno." << std::endl;
    } else {
        std::cout << "Mnozenje matrica je netocno." << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
  
    return 0;  
}

