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

#define ERROR_CHECK(x) if ((x) == NULL) { printf("Failed to allocate memory\n"); exit(EXIT_FAILURE); }

void copyImageFromFile(FILE *srcImage, int *dstImage, int width, int height);
void dumpImageToFile(int *srcImage, char *dstName, int width, int height);
void addSquareToImage(int *srcImage, int width, int height, int position, int x, int y);

int main(int argc, char **argv){
  
  // File I/O set up
  FILE *inputFile, *templateFile;
  char inputFileName[20];
  
  // Performance control
  bool enabledGPU = true;
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
  ERROR_CHECK(image = (int*)malloc(width*height*sizeof(int)));

  ERROR_CHECK(gradientMag = (int*)malloc(width*height*sizeof(int)));

  ERROR_CHECK(gradientDir = (enum direction*)malloc(width*height*sizeof(enum direction)));

  ERROR_CHECK(cannyImage = (int*)malloc(width*height*sizeof(int)));

  ERROR_CHECK(tmplate = (int*)malloc(tWidth*tHeight*sizeof(int)));

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
  
  // Find gradient magnitude and directions
  if (enabledGPU) {    
    calcGradientGPU<<<numBlocks, tPerBlock>>> 
                        (d_image, d_gradientMag, d_gradientDir, width, height, thresh);
    cudaDeviceSynchronize();  

    //TODO: EdgeThinningGPU

    //TODO: ConnectivityAnalysisGPU

    //TODO: MatchTemplateGPU

    checkCudaErrors(cudaMemcpy(gradientMag, d_gradientMag, width*height*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(gradientDir, d_gradientDir, width*height*sizeof(int), cudaMemcpyDeviceToHost));
  }
  else{
    calcGradientCPU(image, gradientMag, gradientDir, width, height, thresh);
  }


  dumpImageToFile(gradientMag, "out-gradient.pgm", width, height);
  int j =0;
  for(i=0; i < height; i ++){
    for (j=0; j <width; j++){
      printf("%d ", gradientDir[i*width + j]);
    }
    printf("\n");
  }
  exit(0);

  // Thin edges using non-maximum suppression
  thinEdgesCPU(gradientMag, gradientDir, width, height);
  dumpImageToFile(gradientMag, "out-edgethin.pgm", width, height);

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


  free(image);
  free(gradientMag);
  free(gradientDir);
  free(cannyImage);
  free(tmplate);

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
