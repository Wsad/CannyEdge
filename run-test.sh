#!/bin/bash

rm gpu_results.txt
rm cpu_results.txt

touch gpu_results.txt
touch cpu_results.txt

template=Simple.pgm
base=Hotel
size=(256 512 1024 2048 4000)

for s in ${size[*]};
do
  for i in `seq 0 5`;
  do
    base=Hotel$s.pgm
    pwd
    "./canny" -f $base -t "$template" --gpu -cLo 45 -cHi 60 >> gpu_results.txt
    "./canny" -f $base -t "$template" --cpu -cLo 45 -cHi 60 >> cpu_results.txt
  done  
done
