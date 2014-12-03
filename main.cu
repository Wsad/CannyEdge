#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "cannyCPU.h"
#include "cannyGPU.cuh"

#define MAXPIXEL 255
#define START_RUN 0
#define GRAD_END  1
#define THIN_END  2
#define CONNECT_END 3
#define TEMPLATE_END 4

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
  bool enabledGPU = false;
  bool enabledCPU = false;
  bool outFiles = false;
  
  // Image information
  int width, height, maxPValue;
  int tWidth, tHeight, *matchedPos;

  int i;

  // Host data items
  int *image, *tmplate;
  int *gradientMag,*cannyImage,thresh = 50;
  enum direction *gradientDir;
  int loThresh = 85;
  int hiThresh = 125;
  bool print = false;

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
    else if (strcmp(argv[i],"--print") == 0){
      print = true;
    }
    else if (strcmp(argv[i],"--files") == 0){
      outFiles =true;
    }
    else if (strcmp(argv[i],"--gpu") == 0){
      enabledGPU = true;
    }
    else if (strcmp(argv[i],"--cpu") == 0){
      enabledCPU = true;
    }
    else if (strcmp(argv[i],"-cLo") == 0){
      loThresh = atoi(argv[i+1]);
    }
    else if (strcmp(argv[i],"-cHi") ==0){
      hiThresh = atoi(argv[i+1]);
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

  int *gpuMag, *gpuMagSuppressed;
  gpuMag = (int*)malloc(sizeof(int)*width*height);
  checkNotNull(gpuMag);
  
  gpuMagSuppressed = (int*)malloc(sizeof(int)*width*height);
  checkNotNull(gpuMag);

  // File I/O 
  // Copy image from file
  copyImageFromFile(inputFile, image, width, height);
  copyImageFromFile(templateFile, tmplate, tWidth, tHeight);

  int numEvents = 5;
  struct timespec eventCPU[numEvents];
  cudaEvent_t eventGPU[numEvents];
  for(int i=0; i<numEvents; i++){
    cudaEventCreate(&eventGPU[i]);
  }
  
  // Find gradient magnitude and directions
  if (enabledGPU) 
  {    
    // Copy image to from host to device
    checkCudaErrors(cudaMemcpy(d_image, image, width*height*sizeof(int), cudaMemcpyHostToDevice));

    int blockDimX = 16;
    int blockDimY = 16;
    int numBlocksX = (width+blockDimX-1)/blockDimX;
    int numBlocksY = (height+blockDimY-1)/blockDimY;
    dim3 tPerBlock(blockDimX,blockDimY);
    dim3 numBlocks(numBlocksX,numBlocksY);
    
    cudaEventRecord(eventGPU[START_RUN]);
    calcGradientGPU<<<numBlocks, tPerBlock>>> 
                     (d_image, d_gradientMag, d_gradientDir, width, height, thresh);

    cudaEventRecord(eventGPU[GRAD_END]);
    cudaEventSynchronize(eventGPU[GRAD_END]);
    
    if(print || outFiles){
      checkCudaErrors(cudaMemcpy(gpuMag, d_gradientMag, width*height*sizeof(int), cudaMemcpyDeviceToHost));
    }
    if(print){
      printf("Initial gradient Magnitude\n");
      printImageASCII(gpuMag, width, height);
    }
    if (outFiles){
      dumpImageToFile(gpuMag, "out-gradientGPU.pgm",width, height);
    }

    thinEdgesGPU<<<numBlocks, tPerBlock>>>(d_gradientMag, d_gradientDir, width, height);
    
    cudaEventRecord(eventGPU[THIN_END]);
    cudaEventSynchronize(eventGPU[THIN_END]);
   
    if(print || outFiles){
      checkCudaErrors(cudaMemcpy(gpuMag, d_gradientMag, width*height*sizeof(int), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(gradientDir, d_gradientDir, width*height*sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (print){
      printf("Thinned GPU\n");
      printImageASCII(gpuMag, width, height);
    }
    if(outFiles){
      dumpImageToFile(gpuMag,"out-thinnedGPU.pgm",width,height);
    }

    hysteresisGPU(d_gradientMag, loThresh, hiThresh, width, height, NULL);
    
    cudaEventRecord(eventGPU[CONNECT_END]);
    cudaEventSynchronize(eventGPU[CONNECT_END]);
    
    if(print){
      checkCudaErrors(cudaMemcpy(gpuMagSuppressed, d_gradientMag, width*height*sizeof(int), cudaMemcpyDeviceToHost));
      printf("Hysteresis\n");
      printImageASCII(gpuMagSuppressed, width, height);
    }
  
    //TODO: MatchTemplateGPU
    cudaEventRecord(eventGPU[TEMPLATE_END]);
    cudaEventSynchronize(eventGPU[TEMPLATE_END]);    

    float time,total;
    printf("STAGE\tGPU Time (ms)\t%dx%d\n",width,height);
    cudaEventElapsedTime(&total,eventGPU[0],eventGPU[numEvents-1]);
    printf("T\t%f\n",numEvents,total);
    for(int i=0; i<numEvents-1; i++){
      cudaEventElapsedTime(&time,eventGPU[i], eventGPU[i+1]);
      printf("%d\t%f\t%f%\n",i,time,100*time/total);
    }
    printf("\n");
    
  }
  if (enabledCPU){
    clock_gettime(CLOCK_MONOTONIC,&eventCPU[START_RUN]);
    calcGradientCPU(image, gradientMag, gradientDir, width, height, thresh);
    clock_gettime(CLOCK_MONOTONIC,&eventCPU[GRAD_END]);
    
    if(outFiles){
      dumpImageToFile(gradientMag, "out-gradient.pgm", width, height);
    }
    
    if(print){
      printf("Grad Magnitude CPU:\n");
      printImageASCII(gradientMag, width, height);
      //if(!arrayMatch(gradientMag, gpuMag, width*height)){
      //printf("CPU gradient magnitude does not match GPU\n");
      //exit(0);
    }

    // Thin edges using non-maximum suppression
    thinEdgesCPU(gradientMag, gradientDir, width, height);
    clock_gettime(CLOCK_MONOTONIC,&eventCPU[THIN_END]);

    if(outFiles){
      dumpImageToFile(gradientMag, "out-edgethin.pgm", width, height);
    }

    if(print){
      printf("Suppressed Edges CPU:\n");
      printImageASCII(gradientMag, width, height);
      //if(!arrayMatch(gradientMag, gpuMagSuppressed, width*height)){
        //printf("CPU suppressed gradient does not match GPU\n");
        //exit(0);
      //}
    }

    // Double Threshold Hysteresis
    connectivityCPU(gradientMag, cannyImage, width, height, loThresh, hiThresh);
    clock_gettime(CLOCK_MONOTONIC,&eventCPU[CONNECT_END]);

    if(outFiles){
      dumpImageToFile(cannyImage, "out-connected.pgm", width, height);
    }
    
    if(print){
      printf("CPU Connected:\n");
      printImageASCII(cannyImage,width,height);
    }

    // TODO: Matching algorithms
    //        Template: Sum of absolute differences, (maybe) Geometric differences
    matchedPos = (int*)malloc(sizeof(int));
    templateMatchCPU(cannyImage, width, height, tmplate, tWidth, tHeight, matchedPos);
    clock_gettime(CLOCK_MONOTONIC,&eventCPU[TEMPLATE_END]);
    if (matchedPos > 0) {
      addSquareToImage(cannyImage, width, height, *matchedPos, tWidth, tHeight);
    }

    long long startMS = eventCPU[START_RUN].tv_sec*1000 + (double)(eventCPU[START_RUN].tv_nsec)/1000000.0;
    long long stopMS = eventCPU[TEMPLATE_END].tv_sec*1000 + (double)(eventCPU[TEMPLATE_END].tv_nsec)/1000000.0;
    double time,total;    
    printf("STAGE\tCPU Time (ms)\t%dx%d\n",width,height);
    total = (double)(stopMS) - (double)startMS;
    printf("T\t%f\n",total);
    for(int i=0; i<numEvents-1; i++){
      startMS = 1000*eventCPU[i].tv_sec + (double)(eventCPU[i].tv_nsec)/1000000.0;
      stopMS = 1000*eventCPU[i+1].tv_sec + (double)(eventCPU[i+1].tv_nsec)/1000000.0;
      time = (double)stopMS - (double)startMS;
      printf("%d\t%f\t%f%\n",i,time,100*time/total);
    }
    printf("\n");
    
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
      printf("%d\t",image[i*width + j]);
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
