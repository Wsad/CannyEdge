#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Image processing operations
#include "cannyCPU.h"
#include "cannyGPU.h"

#define MAXPIXEL 255

#define checkNotNull(x) if ((x) == NULL) { printf("Failed to allocate memory at line:%d\n",__LINE__); exit(EXIT_FAILURE); }

void copyImageFromFile(FILE *srcImage, int *dstImage, int width, int height);
void dumpImageToFile(int *srcImage, char *dstName, int width, int height);
void addSquareToImage(int *srcImage, int width, int height, int position, int x, int y);
void printImageASCII(int *image, int width, int height);
bool arrayMatch(int *a, int *b, int size);

int main(int argc, char **argv){
  
  // File I/O set up
  FILE *inputFile, *templateFile;
  char inputFileName[20];
  
  // Performance control
  bool enabledGPU = true;
  bool enabledCPU = true;
  int threadsPerBlock = 256;
  
  // Image information
  int width, height, maxPValue;
  int tWidth, tHeight, *matchedPos;

  int i;

  // Host data items
  int *image, *tmplate;
  int *gradientMag,*cannyImage,thresh = 50;
  enum direction *gradientDir;

  // Device data items
  int *d_image, *d_gradientMag, *d_gradientDir;
  
  if (argc < 5){
    printf("Incorrect command line arguments\n");
    exit(0);
  }

  for(i=1; i < argc; i++)
  {
    if (strcmp(argv[i],"-f") == 0)
    {
      strcpy(inputFileName,argv[i+1]);
    }
    else if (strcmp(argv[i],"-gThresh") == 0)
    {
      thresh = atoi(argv[i+1]);
    }
    else if (strcmp(argv[i],"-t") == 0)
    {
      templateFile = fopen(argv[i+1],"r");
    }
  }

  inputFile = fopen(inputFileName,"r");

  fscanf(inputFile,"P2 #%*[^\n]\n%d %d %d", &width, &height, &maxPValue);
  if (maxPValue != MAXPIXEL) {
    printf("Incorrect max pixel value\n");
    exit(0);
  }

  fscanf(templateFile,"P2 #%*[^\n]\n%d %d %d", &tWidth, &tHeight, &maxPValue);
  if (maxPValue != MAXPIXEL) {
    printf("Incorrect max pixel value\n");
    exit(0);
  }

  // Allocate Items in Host Memory
  image = (int*)malloc(width*height*sizeof(int));
  checkNotNull(image);

  gradientMag = (int*)malloc(width*height*sizeof(int));
  checkNotNull(gradientMag);

  gradientDir = (enum direction*)malloc(width*height*sizeof(enum direction));
  checkNotNull(gradientDir);

  cannyImage = (int*)malloc(width*height*sizeof(int));
  checkNotNull(cannyImage);

  tmplate = (int*)malloc(tWidth*tHeight*sizeof(int));
  checkNotNull(tmplate);

  // Allocate Items in Device Memory
  checkCudaErrors(cudaMalloc((void**) &d_image, width*height*sizeof(int)));
  
  checkCudaErrors(cudaMalloc((void**) &d_gradientMag, width*height*sizeof(int)));

  checkCudaErrors(cudaMalloc((void**) &d_gradientDir, width*height*sizeof(int)));

  // Copy image from file
  copyImageFromFile(inputFile, image, width, height);
  copyImageFromFile(templateFile, tmplate, tWidth, tHeight);

  // Copy image to from host to device
  checkCudaErrors(cudaMemcpy(d_image, image, width*height*sizeof(int), cudaMemcpyHostToDevice));

  // TODO: proper thread management
  int numBlocks = (width*height + threadsPerBlock -1 )/threadsPerBlock;
  dim3 tPerBlock(16,16);

  // Potential TODO: Noise reduction ( Gaussian )
  
  int *gpuMag, *gpuMagSuppressed;
  gpuMag = (int*)malloc(sizeof(int)*width*height);
  checkNotNull(gpuMag);
  gpuMagSuppressed = (int*)malloc(sizeof(int)*width*height);
  checkNotNull(gpuMag);

  // Find gradient magnitude and directions
  if (enabledGPU) {    
    calcGradientGPU<<<numBlocks, tPerBlock>>> 
                        (d_image, d_gradientMag, d_gradientDir, width, height, thresh);
    cudaDeviceSynchronize();  
    checkCudaErrors(cudaMemcpy(gpuMag, d_gradientMag, width*height*sizeof(int), cudaMemcpyDeviceToHost));
    //printf("Initial gradient Magnitude\n");
    //printImageASCII(gpuMag, width, height);

    thinEdgesGPU<<<numBlocks, tPerBlock>>>(d_gradientMag, d_gradientDir, width, height);
    cudaDeviceSynchronize();
    
    //TODO:hysteresisGPU(d_gradientMag, 80, 170, width, height, null);

    //TODO: MatchTemplateGPU

    
    checkCudaErrors(cudaMemcpy(gpuMagSuppressed, d_gradientMag, width*height*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(gradientDir, d_gradientDir, width*height*sizeof(int), cudaMemcpyDeviceToHost));
    printf("Thinned edges\n");
    printImageASCII(gpuMagSuppressed, width, height);
  }
  if (enabledCPU){
    printImageASCII(image, width, height);
    calcGradientCPU(image, gradientMag, gradientDir, width, height, thresh);
    dumpImageToFile(gradientMag, "out-gradient.pgm", width, height);
    //printf("Grad Magnitude CPU:\n");
    //printImageASCII(gradientMag, width, height);
    /*if(!arrayMatch(gradientMag, gpuMag, width*height)){
      printf("CPU gradient magnitude does not match GPU\n");
      exit(0);
    }*/

    // Thin edges using non-maximum suppression
    printf("Thined CPU:\n");
    thinEdgesCPU(gradientMag, gradientDir, width, height);
    printImageASCII(gradientMag, width, height);
    dumpImageToFile(gradientMag, "out-edgethin.pgm", width, height);
    if(!arrayMatch(gradientMag, gpuMagSuppressed, width*height)){
      printf("CPU suppressed gradient does not match GPU\n");
      exit(0);
    }

    // TODO: Double Threshold (BFS from definite edges over potential edges)
    connectivityCPU(gradientMag, cannyImage, width, height, 85, 125);
    dumpImageToFile(cannyImage, "out-connected.pgm", width, height);

    // TODO: Matching algorithms
    //        Template: Sum of absolute differences, (maybe) Geometric differences
    matchedPos = (int*)malloc(sizeof(int));
    templateMatchCPU(cannyImage, width, height, tmplate, tWidth, tHeight, matchedPos);
    if (matchedPos > 0) {
      addSquareToImage(cannyImage, width, height, *matchedPos, tWidth, tHeight);
    }
    dumpImageToFile(cannyImage, "out-template.pgm", width, height);
  }

  free(image);
  free(gradientMag);
  free(gradientDir);
  free(cannyImage);
  free(tmplate);

  free(gpuMag);
  free(gpuMagSuppressed);

  //Free Device Memory
  cudaFree(d_image);
  cudaFree(d_gradientMag);
  cudaFree(d_gradientDir);

  fclose(inputFile);
  fclose(templateFile);
  return 0;
}

void copyImageFromFile(FILE *srcImage, int *dstImage, int width, int height){
  int i;
  for(i = 0; i < width*height; i++){
    fscanf(srcImage, "%d",&dstImage[i]);
  }
}

void dumpImageToFile(int *srcImage, char *dstName, int width, int height){
  FILE *dstImage = fopen(dstName, "w");
  int i;

  fprintf(dstImage,"P2\n%d %d\n%d\n", width, height, MAXPIXEL);
  for(i =0; i < width*height; i++){
    fprintf(dstImage,"%d\n",srcImage[i]);
  }
  fclose(dstImage);
}

void addSquareToImage(int *srcImage, int width, int height, int position, int x, int y){
  int i,j;
  if (position < 0 || position%width + x > width || position/width > height){
    //out of bounds
  }else{
    for (j=0; j < x; j++){
      if (position%width + j < width) srcImage[position + j] = 255;
      if (position/width + y < height) srcImage[position + y*width + j] = 255;
    }
    for (i=0; i < y; i++){
      if (position%width + x < width) srcImage[position + i*width + x] = 255;
      if (position/width + i < height) srcImage[position + i*width ] = 255;
    }
  }
}

void printImageASCII(int *image, int width, int height){
  int i,j;
  for(i=0; i<height; i++){
    for(j=0; j<width; j++){
      printf("%d ",image[i*width + j]);
    }
    printf("\n");
  }
}

bool arrayMatch(int *a, int *b, int size){
  for(int i=0; i < size; i++){
    if (a[i] != b[i]) return false;
  }
  return true;
}
