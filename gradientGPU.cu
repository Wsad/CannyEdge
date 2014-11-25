#include <math.h>


__global__ void calcGradientGPU(int *image, int *gradientMag, int *gradientDir, int width, int height, int threshold){

  int mask[9] = { -width - 1, -width, -width + 1,
                -1, 0, 1,
                width -1, width, width + 1 };

  int GxMask[9] = { -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1 };

  int GyMask[9] = { 1, 2, 1,
                    0, 0, 0,
                    -1, -2, -1 };

  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( i >= height-1 || j >= width-1) return;
  
  if (i >0 && j > 0){
    int a;
    int Gx = 0;
    int Gy = 0;
    for(a=0; a < 9; a++){
      Gx += GxMask[a]*image[i*width + j + mask[a]];
      Gy += GyMask[a]*image[i*width + j + mask[a]];
    }

    float angle = M_PI/2;
    if (Gx != 0) angle = atan((float)Gy/(float)Gx);
    
    if (angle < 0 ) angle += M_PI;
    gradientDir[i*width + j] = (int)(( angle)*((float)5/(float)M_PI) - 0.1)%4;

    int mag = abs(Gx) + abs(Gy);

    if (mag > threshold){
      gradientMag[i*width + j] = mag;
    }
    else{
      gradientMag[i*width + j] = 0;
    }

  }
  


}
