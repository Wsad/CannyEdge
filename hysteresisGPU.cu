#include <stdio.h>

__global__ void preprocessFirstPass(int *mag,int lo_thres,int hi_thresh,int width,int height);

__device__ void add(int *q, int pos, int front, int back);

__device__ int get(int *q, int front, int back);


#define SHM_DIMX 16
#define Q_SIZE 100

void hysteresisGPU(int *d_mag, int lo_thresh, int hi_thresh, int width, int height, int *testArr){
  
  int threadsPerBlock = 256;
  int blockDimX = 16;
  int blockDimY = 16;
  int numBlocksX = (width+blockDimX-1)/blockDimX;
  int numBlocksY = (height+blockDimY-1)/blockDimY;
  dim3 tPerBlock(16,16);
  dim3 numBlocks(numBlocksX, numBlocksY);


  preprocessFirstPass<<<numBlocks,tPerBlock,sizeof(int)*(blockDimX+1)*(blockDimY+1)>>>
                              (d_mag, lo_thresh, hi_thresh, width, height);

  cudaDeviceSynchronize();


}

__global__ void preprocessFirstPass(int *mag, int lo_thresh, int hi_thresh, int width, int height){
  int d_x = blockIdx.x*blockDim.x + threadIdx.x;
  int d_y = blockIdx.y*blockDim.y + threadIdx.y;

  int s_x = threadIdx.x;
  int s_y = threadIdx.y;

  extern __shared__ int shm[];

  if (d_x > width|| d_y > height) return;
  shm[s_y*SHM_DIMX + s_x] = mag[d_y*width + d_x];
  if (threadIdx.x == 0  && d_x + blockDim.x < width){
    shm[s_y*SHM_DIMX + s_x + blockDim.x] = mag[d_y*width + d_x + blockDim.x];
    int d_startx = blockIdx.x*blockDim.x;
    int d_starty = blockIdx.y*blockDim.y;
    shm[blockDim.y*SHM_DIMX + threadIdx.y] = mag[(d_starty+blockDim.y)*width + d_startx + threadIdx.y];
    shm[blockDim.y*SHM_DIMX + blockDim.x] = mag[(d_starty+blockDim.y)*width + d_startx + blockDim.x];
  }

  __syncthreads();

  int q[Q_SIZE];
  int front = 0;
  int back = 0;

  s_y = s_y +1;
  s_x = s_x +1;

  int i = s_y*SHM_DIMX + s_x;
  int cur = shm[i];
  if (cur > lo_thresh){
    if (cur > hi_thresh){
      shm[i] = -2; //Definite Edge
      add(q,i,front,back);
      //printf("Definite Edge At %d %d\n", d_x+1,d_y+1);
    }
    else{
      shm[i] = -1; //Potential Edge
    }
  }
  else{
    shm[i] = 0; //Non-Edge
  }
}

__device__ void add(int *q, int pos, int front, int back){
 q[back] = pos;
 back = (back+1)%Q_SIZE;
}

__device__ int get(int *q, int front, int back){
  int t = front;
  front = (front+1)%Q_SIZE;
  return q[t];
}
