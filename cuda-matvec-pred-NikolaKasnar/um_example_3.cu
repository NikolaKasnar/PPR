#include <iostream>
// sum mogu dohvatiti na oba sustava
__device__ __managed__ int sum = 0;

struct Data{
    int size;
    int * data;
};

__global__
void kernel(Data d, int * sum){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<d.size)
        atomicAdd(sum, d.data[i]);
} 


int main()
{ 
    Data d;
    d.size = 38;
    cudaMallocManaged(&(d.data), d.size*sizeof(int));
    for(int i=0; i<d.size; ++i)
        d.data[i] = i;

    kernel<<<1,128>>>(d, &sum);
    cudaDeviceSynchronize();
    std::cout << "sum = " << sum << " (= " << (d.size)*(d.size-1)/2 << ")\n";
    return 0;
}