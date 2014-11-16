#include <stdlib.h>
#include <stdio.h>

void templateMatchCPU(int *cannyImage, int width, int height, int *template, int tWidth, int tHeight, int *maxPos){
  int max = -1;

  for (int i=0; i < height - tHeight; i++){
    for (int j=0; j < width - tWidth; j++){
      int sum = 0;
      for (int iOff=0; iOff < tHeight; iOff++){
        for (int jOff=0; jOff < tWidth; jOff++){
          sum += abs(cannyImage[i*width + j + iOff*width + jOff] - template[iOff*tWidth + jOff]);
        }
      }
      if (sum > max){
        max = sum;
        *maxPos = i*width + j;
      }
    }
  }
  if (max < 0) *maxPos = -1;
}