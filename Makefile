CXX = /usr/bin/g++
NVCC = /usr/local/cuda/bin/nvcc

.PHONY: all clean

all: knn_finder

knn_finder: knn_finder.cpp csv_handler.cpp csv_handler.hpp mp_handler.cpp mp_handler.hpp file_handler.cpp file_handler.hpp knn_cuda.cu knn_cuda.hpp
	$(NVCC) -o $@ $< csv_handler.cpp mp_handler.cpp file_handler.cpp knn_cuda.cu -O2 -lcuda -lcublas -lboost_regex -lboost_iostreams -lboost_program_options -lmsgpack

clean:
	rm knn_finder
