#include "cannyCPU.h"

void thinEdgesCPU(int *mag, enum direction *dir, int width, int height){
  for(int i=1; i < height-1; i++)
  {
    for(int j=1; j < width-1; j++)
    {
      int id = i*width + j;
      int dx1, dy1, dx2, dy2;
      switch (dir[id]){
        case WE :
          dx1 = -1;
          dy1 = 0;

          dx2 = 1;
          dy2 = 0;
          break;
        
        case NE :
          dx1 = -1;
          dy1 = 1;

          dx2 = 1;
          dy2 = -1;
          break;

        case N :
          dx1 = 0;
          dy1 = -1;

          dx2 = 0;
          dy2 = 1;
          break;

        case NW :
          dx1 = -1;
          dy1 = -1;

          dx2 = 1;
          dy2 = 1;
          break;
      }

      int curMag = mag[id];

      if ( (mag[(i+dy1)*width + j + dx1] > curMag) || (mag[(i+dy2)*width + j + dx2] > curMag) )
      {
        mag[id] = 0;
      }
    }
  }
}