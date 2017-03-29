#include <iostream>

#include "knn_cuda.hpp"

namespace knn_finder {

__global__ void normalise(float *mat, int pitch, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < width) {
		float val, sum = 0;

		for (int i = 0; i < height; i++) {
			val = mat[i*pitch+x];
			sum += val * val;
		}

		if (sum != 0) {
			sum = rsqrtf(sum);

			for (int i = 0; i < height; i++) {
				mat[i*pitch+x] *= sum;
			}
		}
	}
}

template <typename T> __device__ inline void swap(T *vec, int vec_pitch, int a, int b) {
	T tmp = vec[a*vec_pitch];
	vec[a*vec_pitch] = vec[b*vec_pitch];
	vec[b*vec_pitch] = tmp;
}

__global__ void sort(float *dist, int dist_pitch, int *idx, int idx_pitch, int width, int k) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < width) {
		float *dist_ptr = dist + x;
		int *idx_ptr = idx + x;

		for (int i = 0; i < k; i++) idx_ptr[i*idx_pitch] = i;

		int num_phases = 0;
		for (int i = k - 1; i > 0; i >>= 1) num_phases++;

		for (int phase = 0; phase < num_phases; phase++) {
			int chunk_half_size = 1 << phase;
			int chunk_size = chunk_half_size << 1;

			for (int chunk_offset = 0; chunk_offset < k;) {
				for (int i = chunk_half_size - 1; i >= 0; i--) {
					int a = chunk_offset + i;
					int b = chunk_offset + (chunk_size - 1) - i;

					if (b >= k) break;
					if (dist_ptr[a*dist_pitch] > dist_ptr[b*dist_pitch]) {
						swap(dist_ptr, dist_pitch, a, b);
						swap(idx_ptr, idx_pitch, a, b);
					}
				}

				int next_chunk = chunk_offset + chunk_size;
				for (int div_size = chunk_half_size; div_size > 1; div_size >>= 1) {
					int div_half_size = div_size >> 1;

					for (int div_offset = chunk_offset; div_offset < next_chunk; div_offset += div_size) {
						for (int i = 0; i < div_half_size; i++) {
							int a = div_offset + i;
							int b = div_offset + i + div_half_size;

							if (b >= k) break;
							if (dist_ptr[a*dist_pitch] > dist_ptr[b*dist_pitch]) {
								swap(dist_ptr, dist_pitch, a, b);
								swap(idx_ptr, idx_pitch, a, b);
							}
						}
					}
				}
				chunk_offset = next_chunk;
			}
		}
	}
}

__global__ void insert(float *dist, int dist_pitch, int *idx, int idx_pitch, int width, int height, int k, int base) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < width) {
		float *dist_ptr = dist + x;
		int *idx_ptr = idx + x;

		float *max_ptr = &dist_ptr[(k-1)*dist_pitch];
		float max_dist = *max_ptr;
		for (int cur_offset = k; cur_offset < height; cur_offset++) {
			float cur_dist = dist_ptr[cur_offset*dist_pitch];

			if (cur_dist < max_dist) {
				int ins_offset = k - 2;
				for (; ins_offset >= 0; ins_offset--) {
					if (dist_ptr[ins_offset*dist_pitch] <= cur_dist) break;
				}
				ins_offset++;

				for (int i = k - 1; i > ins_offset; i--) {
					dist_ptr[i*dist_pitch] = dist_ptr[(i-1)*dist_pitch];
					idx_ptr[i*idx_pitch] = idx_ptr[(i-1)*idx_pitch];
				}

				dist_ptr[ins_offset*dist_pitch] = cur_dist;
				idx_ptr[ins_offset*idx_pitch] = base + cur_offset;

				max_dist = *max_ptr;
			}
		}
	}
}

__global__ void shift_origin(float *mat, int pitch, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < width) {
		float adj = -mat[x];

		for (int i = 0; i < height; i++) {
			mat[i*pitch+x] = __fadd_rz(mat[i*pitch+x], adj);
		}
	}
}

void find_knn(float *h_mat, int mat_width, int mat_height, int k, float *h_dist, int *h_idx, int is_sparse) {
	int max_width = MAX_WIDTH, max_height = MAX_HEIGHT;

	// Our kernels do not use shared memory explicitly,
	// but "PreferShared" setting performs well empirically.
	// This may be because cuBLAS uses shared memory.
	check_cuda_error(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	float *d_mat;
	size_t mat_byte_pitch;
	check_cuda_error(cudaMallocPitch((void **)&d_mat, &mat_byte_pitch, mat_width * sizeof(float), mat_height), mat_width * sizeof(float) * mat_height);
	size_t mat_pitch = mat_byte_pitch / sizeof(float);

	float *d_dist;
	size_t dist_byte_pitch;
	check_cuda_error(cudaMallocPitch((void **)&d_dist, &dist_byte_pitch, max_width * sizeof(float), max_height + k), max_width * sizeof(float) * (max_height + k));
	size_t dist_pitch = dist_byte_pitch / sizeof(float);

	int *d_idx;
	size_t idx_byte_pitch;
	check_cuda_error(cudaMallocPitch((void **)&d_idx, &idx_byte_pitch, max_width * sizeof(int), k), max_width * sizeof(int) * k);
	size_t idx_pitch = idx_byte_pitch / sizeof(int);

	check_cuda_error(cudaMemcpy2D(d_mat, mat_byte_pitch, h_mat, mat_width * sizeof(float), mat_width * sizeof(float), mat_height, cudaMemcpyHostToDevice));

	dim3 num_blocks((mat_width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	normalise<<<num_blocks, threads_per_block>>>(d_mat, mat_pitch, mat_width, mat_height);

	cublasHandle_t handle;
	check_cublas_error(cublasCreate(&handle));
	const float alpha = -1.0;
	const float beta = 0.0;

	for (int i = 0; i < mat_width; i += max_width) {
		int actual_width = std::min<int>(max_width, mat_width - i);
		dim3 num_slice_blocks((actual_width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);

		for (int j = 0; j < mat_width; j += max_height) {
			int actual_height = std::min<int>(max_height, mat_width - j);

			if (j == 0) {
				check_cublas_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, actual_width, actual_height, mat_height, &alpha, &d_mat[i], mat_pitch, &d_mat[j], mat_pitch, &beta, d_dist, dist_pitch));

				sort<<<num_slice_blocks, threads_per_block>>>(d_dist, dist_pitch, d_idx, idx_pitch, actual_width, k);
				insert<<<num_slice_blocks, threads_per_block>>>(d_dist, dist_pitch, d_idx, idx_pitch, actual_width, actual_height, k, j);
			} else {
				check_cublas_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, actual_width, actual_height, mat_height, &alpha, &d_mat[i], mat_pitch, &d_mat[j], mat_pitch, &beta, &d_dist[k*dist_pitch], dist_pitch));

				insert<<<num_slice_blocks, threads_per_block>>>(d_dist, dist_pitch, d_idx, idx_pitch, actual_width, actual_height + k, k, j - k);
			}
		}

		shift_origin<<<num_slice_blocks, threads_per_block>>>(d_dist, dist_pitch, actual_width, k);

		check_cuda_error(cudaMemcpy2D(&h_dist[i], mat_width * sizeof(float), d_dist, dist_byte_pitch, actual_width * sizeof(float), k, cudaMemcpyDeviceToHost));
		check_cuda_error(cudaMemcpy2D(&h_idx[i], mat_width * sizeof(int), d_idx, idx_byte_pitch, actual_width * sizeof(int), k, cudaMemcpyDeviceToHost));

		std::cout << "Processed " << i + actual_width << " records." << std::endl;
	}

	cublasDestroy(handle);

	cudaFree(d_idx);
	cudaFree(d_dist);
	cudaFree(d_mat);
}

void print_props(int device) {
	cudaDeviceProp prop;
	check_cuda_error(cudaGetDeviceProperties(&prop, device));

	std::cout << "Device name: " << prop.name << std::endl;
	std::cout << "Shader model: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << std::endl;
	std::cout << "Total global memory: " << prop.totalGlobalMem << std::endl;
	std::cout << "Max distance matrix size: " << MAX_WIDTH << " x " << MAX_HEIGHT << std::endl;
	std::cout << "Number of threads per block: " << THREADS_PER_BLOCK << std::endl;
}

void check_cuda_error(cudaError_t res, size_t size) {
	if (res) {
		std::cerr << "CUDA runtime error: " << cudaGetErrorString(res) << std::endl;
		if (size != 0) {
			std::cerr << "Tried to acquire " << size << " bytes of memory." << std::endl;
			exit(CUDA_MALLOC_FAILED);
		}
		exit(CUDA_ERROR);
	}
}

void check_cublas_error(cublasStatus_t res) {
	if (res) {
		std::cerr << "cuBLAS runtime error: " << res << std::endl;
		exit(CUBLAS_ERROR);
	}
}

} // knn_finder
