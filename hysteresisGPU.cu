__global__ void preprocessFirstPass(int *mag,int lo_thres,int hi_thresh,int width,int height);

void hysteresisGPU(int *d_mag, int lo_thresh, int hi_thresh, int width, int height, int *testArr){
  
  int threadsPerBlock = 256;
  int numPixels = width*height;
  int numBlocks = (numPixels + threadsPerBlock -1)/threadsPerBlock;

  preprocessFirstPass<<<numBlocks,threadsPerBlock>>>
                              (d_mag, lo_thresh, hi_thresh, width, height);

  cudaDeviceSynchronize();


}

__global__ void preprocessFirstPass(int *mag, int lo_thresh, int hi_thresh, int width, int height){
  int x = blockIdx.x*blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y*blockDim.y + threadIdx.y + 1;

  if (x > width - 1 || y > height -1 ) return;
  
  int i = y*width + x;
  int cur = mag[i];
  if (cur > lo_thresh){
    if (cur > hi_thresh){
      mag[i] = -2; //Definite Edge
    }
    else{
      mag[i] = -1; //Potential Edge
    }
  }
  else{
    mag[i] = 0; //Non-Edge
  }
  

}
