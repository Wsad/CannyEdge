#include <stdlib.h>
#include <stdio.h>

typedef struct{
  int front,rear,size;
  int *vals;
} queue_t;

void put(queue_t *queue, int pixel);
int get(queue_t *queue);

void connectivityCPU(int *gradient, int *connectedImage, int width, int height, int lowThresh, int hiThresh){
  queue_t queue;
  queue.front = 0;
  queue.rear = 0;
  queue.size = width*height;
  queue.vals = (int*)malloc(width*height*sizeof(int));
  int i,j,iOff,jOff;

  //Initial setup to find starting points for BFS
  for (i=0; i < height; i++)
  {
    for (j=0; j < width; j++)
    {
      if (gradient[i*width + j] > lowThresh)
      {
        if(gradient[i*width + j] > hiThresh)
        {
          //Definite Edge
          connectedImage[i*width + j] = -2;
          put(&queue, i*width + j);
        }
        else
        {
          //Potential Edge
          connectedImage[i*width + j] = -1;
        }
      }
      else
      {
        // Non-edge
        connectedImage[i*width + j] = 0;
      }
    }
  }

  int current = get(&queue);
  while (current > 0)
  {
    for(iOff = -1; iOff < 2; iOff++){
      for(jOff = -1; jOff < 2; jOff++){
        int t = current - iOff - jOff*width;
        if ( (0 <= t) && (t <= width*height) && (connectedImage[t] == -1) )
        {
          //Mark and add potential edge
          connectedImage[t] = -2;
          put(&queue,t);
        }
      }
    }

    current = get(&queue);
  }


  //Convert Enumerations to Image format
  for (i=0; i < height; i++)
  {
    for (j=0; j < width; j++)
    {
      if (connectedImage[i*width + j] == -2)
      {
        //Mark as white pixel
        connectedImage[i*width + j] = 255;
      }
      else
      {
        //Mark as black pixel
        connectedImage[i*width + j] = 0;
      }
    }
  }
}

void put(queue_t *q, int pixel){
  if ((q->rear + 1) % q->size == q->front){
    printf("Error: attempt to put item in full queue\n"); exit(-1);
  } else {
    q->vals[q->rear] = pixel;
    q->rear = (q->rear+1)%q->size;
  }
}

int get(queue_t *q){
  if (q->front == q->rear){
    return -1;
  } else {
    q->front = (q->front+1)%q->size;
    return q->vals[q->front];
  }
}
