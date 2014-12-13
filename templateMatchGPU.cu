#include <stdio.h>
#include <cuda_runtime.h>

__global__ void templateMatchGPU(int *image, int width, int height, int * tmplate, int tWidth, int tHeight, int *result){

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.x*blockDim.y + threadIdx.y;
  
  if (idx > width-1 || idy > height-1) return;

  int sum = 0;
  for(int i =0; i < tHeight; i++){
    for(int j=0; j < tWidth; j++){
      if ( idy+i < width  && idx + j < height){
        sum += abs(image[(idy+i)*width + idx + j] - tmplate[i*tWidth + j]);
      }
    }
  }

  result[idy*width + idx] = sum;
   
}
