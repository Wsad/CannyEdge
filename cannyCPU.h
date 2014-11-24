//cannyCPU.h

enum direction { WE, NE, N, NW };

void calcGradientCPU(int *src, int *dst, enum direction *dir, int width, int height, int threshold);
void thinEdgesCPU(int *mag, enum direction *dir, int width, int height);
void connectivityCPU(int *gradient, int *connectedImage, int width, int height, int lowThresh, int hiThresh);
void templateMatchCPU(int *cannyImage, int width, int height, int *tmplate, int tWidth, int tHeight, int *maxPos);
