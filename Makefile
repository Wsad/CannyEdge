CUDACC=nvcc
CFLAGS= -c 
DEBUG= -arch=sm_20
INCL= -I /usr/local/cuda/samples/common/inc
LINKS=
OBJ=$(patsubst %.cu,%.o ,$(wildcard *.cu))
OBJ+=$(patsubst %.cpp,%.o ,$(wildcard *.cpp))

EXEC=canny
EXEC_DBG=dbg_canny

all: $(EXEC)

debug: $(EXEC_DBG)

$(EXEC) : $(OBJ)
	$(CUDACC) $(LINKS) *.o -o $@

%.o:%.cu
	$(CUDACC) $(INCL) $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CUDACC) $(INCL) $(CFLAGS) $< -o $@

clean:
	rm *.o 
