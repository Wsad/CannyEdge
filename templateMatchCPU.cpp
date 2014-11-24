#include <stdlib.h>
#include <stdio.h>

void templateMatchCPU(int *cannyImage, int width, int height, int *tmplate, int tWidth, int tHeight, int *maxPos){
  int max = -1;
  int i,j,iOff,jOff;

  for (i=0; i < height - tHeight; i++){
    for (j=0; j < width - tWidth; j++){
      int sum = 0;
      for (iOff=0; iOff < tHeight; iOff++){
        for (jOff=0; jOff < tWidth; jOff++){
          sum += abs(cannyImage[i*width + j + iOff*width + jOff] - tmplate[iOff*tWidth + jOff]);
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
