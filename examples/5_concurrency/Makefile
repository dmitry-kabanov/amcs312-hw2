TARGET=toy
SRC=concurrency.cu

all:
	nvcc $(SRC) -arch sm_35 -o $(TARGET) -lcudart

clean:
	rm -f -v $(TARGET)
