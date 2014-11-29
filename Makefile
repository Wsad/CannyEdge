CUDACC=nvcc
CFLAGS= -c -arch=sm_20
INCL= -I /usr/local/cuda/samples/common/inc
LINKS=
OBJ=$(patsubst %.cu,%.o ,$(wildcard *.cu))
OBJ+=$(patsubst %.cpp,%.o ,$(wildcard *.cpp))

EXEC=canny

all: $(EXEC)

$(EXEC) : $(OBJ)
	$(CUDACC) $(LINKS) *.o -o $@

%.o:%.cu
	$(CUDACC) $(INCL) $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CUDACC) $(INCL) $(CFLAGS) $< -o $@

clean:
	rm *.o 
