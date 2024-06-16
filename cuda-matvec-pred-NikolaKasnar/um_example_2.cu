#include <iostream>

struct Data{
    int size;
    int sum;
    int * data;
};

__global__
void kernel(Data & d){ // Kompilira se ali nema prijenosa po referenci
    int i = blockIdx.x*blockDim.x+threadIdx.x; 
    if(i<d.size){
        //printf("d.size = %d ",d.size);
        atomicAdd(&(d.sum), d.data[i]);
        d.data[i]++;
    } 
    __syncthreads();
    // if(i == 0)
    //     printf("d.sum = %d\n",d.sum);  // NE ISPISUJE ???
} 


int main()
{
    // Ovaj primjer ne radi. d.sum i d.size su kopirani na device, ali nisu u unificiranom prostoru
    Data d;
    d.size = 16;
    d.sum = 0;
    cudaMallocManaged(&(d.data), d.size*sizeof(int));
    for(int i=0; i<d.size; ++i)
        d.data[i] = i;

    kernel<<<1,128>>>(d);
    cudaDeviceSynchronize();
    std::cout << "\n Main: sum = " << d.sum << " (= " << (d.size)*(d.size-1)/2 << ")\n";
    std::cout << "d.data[0] = " << d.data[0] << "\n";  // Ne mijenja podatke
    return 0;
}