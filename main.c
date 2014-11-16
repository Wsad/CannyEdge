#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Image processing operations
#include "cannyCPU.h"

#define MAXPIXEL 255

void copyImageFromFile(FILE *srcImage, int *dstImage, int width, int height);
void dumpImageToFile(int *srcImage, char *dstName, int width, int height);
void addSquareToImage(int *srcImage, int width, int height, int position, int x, int y);

int main(int argc, char **argv){
  FILE *inputFile, *templateFile;
  char inputFileName[20];
  int width, height, maxPValue;
  int tWidth, tHeight, *matchedPos;

  int *image, *template;
  int *gradientMag,*cannyImage,thresh = 50;
  enum direction *gradientDir;
  
  if (argc < 5){
    printf("Incorrect command line arguments\n");
    exit(0);
  }

  for(int i=1; i < argc; i++)
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

  if ( (image = (int*)malloc(width*height*sizeof(int))) == 0 ){
    printf("Error allocating image\n");
    exit(0);
  }

  if ( (gradientMag = (int*)malloc(width*height*sizeof(int))) == 0 ){
    printf("Error allocating gradient magnitude array\n");
    exit(0);
  }

  if ( (gradientDir = (enum direction*)malloc(width*height*sizeof(enum direction))) == 0 ){
    printf("Error allocating gradient direction array\n");
    exit(0);
  }

  if ( (cannyImage = (int*)malloc(width*height*sizeof(int))) == 0 ){
    printf("Error allocating cannyImage array\n");
    exit(0);
  }

  if ( (template = (int*)malloc(tWidth*tHeight*sizeof(int))) == 0 ){
    printf("Error allocating template array\n");
    exit(0);
  }

  copyImageFromFile(inputFile, image, width, height);
  copyImageFromFile(templateFile, template, tWidth, tHeight);

  // Potential TODO: Noise reduction ( Gaussian )
  
  // Find gradient magnitude and directions
  calcGradientCPU(image, gradientMag, gradientDir, width, height, thresh);
  dumpImageToFile(gradientMag, "out-gradient.pgm", width, height);

  // Thin edges using non-maximum suppression
  thinEdgesCPU(gradientMag, gradientDir, width, height);
  dumpImageToFile(gradientMag, "out-edgethin.pgm", width, height);

  // TODO: Double Threshold (BFS from definite edges over potential edges)
  connectivityCPU(gradientMag, cannyImage, width, height, 85, 125);
  dumpImageToFile(cannyImage, "out-connected.pgm", width, height);

  // TODO: Matching algorithms
  //        Template: Sum of absolute differences, (maybe) Geometric differences
  matchedPos = (int*)malloc(sizeof(int));
  templateMatchCPU(cannyImage, width, height, template, tWidth, tHeight, matchedPos);
  if (matchedPos > 0) {
    addSquareToImage(cannyImage, width, height, *matchedPos, tWidth, tHeight);
  }
  dumpImageToFile(cannyImage, "out-template.pgm", width, height);


  free(image);
  free(gradientMag);
  free(gradientDir);
  free(cannyImage);
  free(template);
  fclose(inputFile);
  fclose(templateFile);
  return 0;
}

void copyImageFromFile(FILE *srcImage, int *dstImage, int width, int height){
  for(int i = 0; i < width*height; i++){
    fscanf(srcImage, "%d",&dstImage[i]);
  }
}

void dumpImageToFile(int *srcImage, char *dstName, int width, int height){
  FILE *dstImage = fopen(dstName, "w");
  fprintf(dstImage,"P2\n%d %d\n%d\n", width, height, MAXPIXEL);
  for(int i =0; i < width*height; i++){
    fprintf(dstImage,"%d\n",srcImage[i]);
  }
  fclose(dstImage);
}

void addSquareToImage(int *srcImage, int width, int height, int position, int x, int y){
  if (position < 0 || position%width + x > width || position/width > height){
    //out of bounds
  }else{
    for (int j=0; j < x; j++){
      if (position%width + j < width) srcImage[position + j] = 255;
      if (position/width + y < height) srcImage[position + y*width + j] = 255;
    }
    for (int i=0; i < y; i++){
      if (position%width + x < width) srcImage[position + i*width + x] = 255;
      if (position/width + i < height) srcImage[position + i*width ] = 255;
    }
  }
}