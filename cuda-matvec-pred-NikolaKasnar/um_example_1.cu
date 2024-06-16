#include <iostream>

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
    int * psum;
    cudaMallocManaged(&psum, sizeof(int));
    Data d;
    d.size = 28;
    cudaMallocManaged(&(d.data), d.size*sizeof(int));
    for(int i=0; i<d.size; ++i)
        d.data[i] = i;

    kernel<<<1,128>>>(d, psum);
    cudaDeviceSynchronize();
    std::cout << "sum = " << *psum << " (= " << (d.size)*(d.size-1)/2 << ")\n";
    return 0;
}