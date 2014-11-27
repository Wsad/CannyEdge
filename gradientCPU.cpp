#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cannyCPU.h"

#define min(a,b) (a > b)? b : a;

// TODO: Add threshold parameter
void calcGradientCPU(int *src, int *mag, enum direction *dir, int width, int height, int threshold){
  int i,j,iOff,jOff;

  int gxMask[] = {  -1, 0, 1, 
                    -2, 0, 2,
                    -1, 0, 1  };
  
  int gyMask[] = {   1,  2, 1,
                     0,  0, 0,
                    -1, -2, -1 };

  // Border values skipped for now
  // TODO: initial loop for edge pixels (with extend or wrap)
  for(i=1; i < height-1; i++)
  {
    for(j=1; j < width-1; j++)
    {
      int Gx = 0;
      int Gy = 0;
      // X Gradient
      for(iOff = -1; iOff < 2; iOff++){
        for(jOff = -1; jOff < 2; jOff++){
          Gx += gxMask[(iOff+1)*3 + jOff+1]*src[(i+iOff)*width + j + jOff];
        }
      }

      // Y Gradient
      for(iOff = -1; iOff < 2; iOff++){
        for(jOff = -1; jOff < 2; jOff++){
          Gy += gyMask[(iOff+1)*3 + jOff+1]*src[(i+iOff)*width + j + jOff];
        }
      }

      int result = abs(Gx) + abs(Gy);
      if (result > threshold){
        mag[i*width + j] = min(result/6, 255);
      } else {
        mag[i*width + j] = 0;
      }

      double angle = 90;
      if(Gx != 0){
        angle = (180.0/3.14159265)*atan(((double)Gy)/((double)Gx));
      } 

      //if (i == 10) printf ("i:%d\tGx:%d\tGy:%d\tAngle:%f\n",i,Gx,Gy,angle);

      if (angle > 22.5 && angle <= 67.5 ){
        dir[i*width + j] = NE;
      }
      else if ((angle > 67.5 && angle <= 90) || (-90 <= angle &&  angle < -67.5)){
        dir[i*width + j] = N;
      }
      else if (-67.5 <= angle && angle < -22.5){
        dir[i*width + j] = NW;
      }
      else {
        dir[i*width + j] = WE;
      }
    }
  }
}
