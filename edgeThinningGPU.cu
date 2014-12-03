

__global__ void thinEdgesGPU(int *mag, int *dir, int width, int height){
  
  int y = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int x = blockIdx.x*blockDim.x + threadIdx.x + 1;

  // Check whether thread is within image boundary
  if (x > width-2 || y > height-2) return;

  // Get gradient direction for current thread
  int tDir = dir[y*width + x];
  
  // Transform offsets so we can find adjacent pixels in direction of gradient mag
  // 0 <= dir <= 3                3   2   1
  //                              0   P   0
  //                              1   2   3
  int xOff = 1;
  int yOff = 0;
  if (tDir > 0){
    xOff = tDir - 2;
    yOff = 1;
  }

  int adjPixel1 = mag[ (y+yOff)*width + x + xOff ];
  int curPixel  = mag[ y*width + x ];
  int adjPixel2 = mag[ (y-yOff)*width + x - xOff ];


  if ( adjPixel1 > curPixel || adjPixel2 > curPixel){
    mag[y*width + x] = 0;
  }
}

  
