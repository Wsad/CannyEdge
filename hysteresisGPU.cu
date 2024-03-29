#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

__global__ void hysteresisFirstPass(int *mag,int lo_thres,int hi_thresh,int width,int height);

__global__ void hysteresisSecondPass(int *mag,int lo_thres,int hi_thresh,int width,int height);


__device__ void add(int *q, int pos, int *front, int *back);

__device__ int get(int *q, int *front, int *back);


#define TILE_DIM 8
#define Q_SIZE 100

void hysteresisGPU(int *d_mag, int lo_thresh, int hi_thresh, int width, int height, int *testArr){
  
  cudaEvent_t pass1,pass2;
  cudaEventCreate(&pass1);
  cudaEventCreate(&pass2);
  
  int blockDimX = 6;//Must be TILE_DIM - 2
  int blockDimY = 6;//Must be TILE_Dim - 2
  int numBlocksX = (width+TILE_DIM-1)/TILE_DIM;
  int numBlocksY = (height+TILE_DIM-1)/TILE_DIM;
  dim3 tPerBlock(blockDimX,blockDimY);
  dim3 numBlocks(numBlocksX, numBlocksY);

  hysteresisFirstPass<<<numBlocks,tPerBlock,sizeof(int)*TILE_DIM*TILE_DIM>>>
                              (d_mag, lo_thresh, hi_thresh, width, height);

  cudaEventRecord(pass1);
  cudaEventSynchronize(pass1);

  tPerBlock.x = TILE_DIM;
  tPerBlock.y = 4;
  hysteresisSecondPass<<<numBlocks,tPerBlock, sizeof(int)*TILE_DIM*TILE_DIM>>>
                              (d_mag, lo_thresh, hi_thresh, width, height);

  cudaEventRecord(pass2);
  cudaEventSynchronize(pass2);                              

}

__global__ void hysteresisFirstPass(int *mag, int lo_thresh, int hi_thresh, int width, int height){
  int d_x = blockIdx.x*TILE_DIM + threadIdx.x;
  int d_y = blockIdx.y*TILE_DIM + threadIdx.y;

  int tidX = threadIdx.x;
  int tidY = threadIdx.y;

  extern __shared__ int sm[];

  // Copy tile in to shared memory
  for (int i=0; i < TILE_DIM; i += blockDim.y){
    for(int j=0; j < TILE_DIM; j += blockDim.x){
      if ((d_x+j)<width && (d_y+i)<height && (tidX+j)<TILE_DIM && (tidY+i)<TILE_DIM){
        sm[(tidY+i)*TILE_DIM + tidX+j] = mag[(d_y+i)*width + d_x+j];
      }
    }
  }

  __syncthreads();


  int q[Q_SIZE];
  int front = 0;
  int back = 0;

  int i = (threadIdx.y+1)*TILE_DIM + threadIdx.x+1;
  int cur = sm[i];
  if (cur > lo_thresh){
    if (cur > hi_thresh){
      sm[i] = -2; //Definite Edge
      add(q,i,&front,&back);
    }
    else{
      sm[i] = -1; //Potential Edge
    }
  }
  else{
    sm[i] = 0; //Non-Edge
  }

  __syncthreads();

  // Thread does its 'walk' or BFS
  while(front != back){
    i = get(q,&front,&back);
    
    for(int yOff=-TILE_DIM; yOff <= TILE_DIM; yOff += TILE_DIM){
      for(int o=yOff-1; o < yOff+2; o++){
        if((i+o)%TILE_DIM < TILE_DIM && (i+o)/TILE_DIM < TILE_DIM){
          if(sm[i+o] == -1){
            sm[i+o] = -2;
            add(q,i+o,&front,&back);
          }
        }
      }
    }
    
    __syncthreads();
  }
  __syncthreads();

  // Copy tile to device memory
  for (int i=0; i < TILE_DIM; i += blockDim.y){
    for(int j=0; j < TILE_DIM; j += blockDim.x){
      if ((d_x+j)<width && (d_y+i)<height && (tidX+j)<TILE_DIM && (tidY+i)<TILE_DIM){
        mag[(d_y+i)*width + d_x+j] = sm[(tidY+i)*TILE_DIM + tidX+j];
      }
    }
  }
  
}

__global__ void hysteresisSecondPass(int *mag, int lo_thresh, int hi_thresh, int width, int height){
  int d_x = blockIdx.x*TILE_DIM + threadIdx.x;
  int d_y = blockIdx.y*TILE_DIM + threadIdx.y;

  int tidX = threadIdx.x;
  int tidY = threadIdx.y;

  extern __shared__ int sm[];

  //Copy tile to shared memory (only use threads first 3 rows)
  if (tidY < 3){
    for(int i=0; i < TILE_DIM; i += blockDim.y-1){
      if(tidX < TILE_DIM && (tidY+i < TILE_DIM) && (d_x < width) && (d_y + i < height)){
        sm[(tidY+i)*TILE_DIM + tidX] = mag[(d_y+i)*width + d_x]; 
      }
    }
  }
  __syncthreads();

  
  // Set up thread indices within shared memory (only apron pixels)
  int x = tidX;
  int y = tidY;
  int xOffset = 0;
  int yOffset = -1;
  if (tidY == 1){
    y = TILE_DIM-1;
    yOffset = 1;
  }else if (tidY > 1){
    int col = tidX + (tidY-2)*blockDim.x;
    x = (col%2)*(TILE_DIM-1);
    y = col/2;

    xOffset = (col%2)*2 -1;
    yOffset = 0;
  }

  int tX = blockIdx.x*TILE_DIM;
  int tY = blockIdx.y*TILE_DIM;

  if(xOffset !=0 &&  (tX +x + xOffset < width) && (tX+x+xOffset > 0) && (tY+y < height)){
    if (mag[(tY+y)*width + tX + x + xOffset] > hi_thresh){
      sm[y*TILE_DIM +x] = -2;
      // Add to queue //add(y*TILE_DIM + x);
    }
  }
  if (yOffset != 0 && (tX + x < width) && (tY+y+yOffset > 0) && (tY+y+yOffset < height)){
    if (mag[(tY+y+yOffset)*width + tX + x] > hi_thresh){
      sm[y*TILE_DIM +x] = -2;
      // Add to queue //add(y*TILE_DIM + x);
    }
  }
  
  __syncthreads();

  // Perform BFS with updated apron pixels
  int q[Q_SIZE];
  int front = 0;
  int back = 0;

  while(front != back){
    int i = get(q,&front,&back);
    
    for(int yOff=-TILE_DIM; yOff <= TILE_DIM; yOff += TILE_DIM){
      for(int o=yOff-1; o < yOff+2; o++){
        if((i+o)%TILE_DIM < TILE_DIM && (i+o)/TILE_DIM < TILE_DIM){
          if(sm[i+o] == -1){
            sm[i+o] = -2;
            add(q,i+o,&front,&back);
          }
        }
      }
    }
    
    __syncthreads();
  }
  __syncthreads();
  
  
  // Copy results to global memory
  if (tidY < 3){
    for(int i=0; i < TILE_DIM; i += blockDim.y-1){
      if(tidX < TILE_DIM && (tidY+i < TILE_DIM) && (d_x < width) && (d_y + i < height)){
        mag[(d_y+i)*width + d_x]  = sm[(tidY+i)*TILE_DIM + tidX]; 
      }
    }
  }

}

__device__ void add(int *q, int pos, int *front, int *back){
 q[*back] = pos;
 *back = (*back+1)%Q_SIZE;
}

__device__ int get(int *q, int *front, int *back){
  int t = *front;
  *front = (*front+1)%Q_SIZE;
  return q[t];
}
