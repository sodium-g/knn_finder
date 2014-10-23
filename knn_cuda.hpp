#ifndef KNN_CUDA_HPP
#define KNN_CUDA_HPP

#include "cuda.h"
#include "cublas_v2.h"

#define CUDA_ERROR 10
#define CUDA_MALLOC_FAILED 11
#define CUBLAS_ERROR 20

#ifndef MAX_WIDTH
#define MAX_WIDTH 32768
#endif
#ifndef MAX_HEIGHT
#define MAX_HEIGHT 4096
#endif
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

namespace knn_finder {

void find_knn(float *h_mat, int mat_width, int mat_height, int k, float *h_dist, int *h_idx);

void print_props(int device=0);

void check_cuda_error(cudaError_t res, size_t size=0);
void check_cublas_error(cublasStatus_t res);

} // knn_finder

#endif // KNN_CUDA_HPP
