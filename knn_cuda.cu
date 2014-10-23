#include <iostream>

#include "knn_cuda.hpp"

namespace knn_finder {

void find_knn(float *h_mat, int mat_width, int mat_height, int k, float *h_dist, int *h_idx) {
}

void print_props(int device) {
	cudaDeviceProp prop;
	check_cuda_error(cudaGetDeviceProperties(&prop, device));

	std::cout << "Device name: " << prop.name << std::endl;
	std::cout << "Shader model: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << std::endl;
	std::cout << "Total global memory: " << prop.totalGlobalMem << std::endl;
	std::cout << "Max distance matrix size: " << MAX_HEIGHT << " x " << MAX_WIDTH << std::endl;
	std::cout << "Number of threads per block: " << THREADS_PER_BLOCK << std::endl;
}

void check_cuda_error(cudaError_t res, size_t size) {
	if (res) {
		std::cerr << "CUDA runtime error: " << cudaGetErrorString(res) << std::endl;
		if (size != 0) {
			std::cerr << "Tried to acquire " << size << " bytes of memory." << std::endl;
			exit(CUDA_MALLOC_FAILED);
		} else {
			exit(CUDA_ERROR);
		}
	}
}

void check_cublas_error(cublasStatus_t res) {
	if (res) {
		std::cerr << "cuBLAS runtime error: " << res << std::endl;
		exit(CUBLAS_ERROR);
	}
}

} // knn_finder
