
__global__ void calcGradientGPU(int *image, int *gradientMag, int *gradientDir, int width, int height, int threshold);

__global__ void thinEdgesGPU(int *mag, int *dir, int width, int height);

void hysteresisGPU(int *d_mag, int lo_thresh, int hi_thresh, int width, int height, int *testArr);
