



__global__ void calcGradientGPU(int *image, int *gradientMag, int *gradientDir, int width, int height){

  int mask[9] = { -width - 1, -width, -width + 1,
                -1, 0, 1,
                width -1, width, width + 1 };

  int pos = blockIdx.x*blockDim.x + threadIdx.x;


}
