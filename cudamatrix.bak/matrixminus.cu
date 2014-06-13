#include "matrix.h"
#include "externs.h"

// ”нарный минус матрицы с использованием global пам€ти
template<class T> __global__ void __global__matrix_minus__global__(T *a, T *b, int height, int width, size_t pitch1, size_t pitch2) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			ELEMENT(T,b,i,j,pitch2) = -ELEMENT(T,a,i,j,pitch1);
		}
	}
}

// ”нарный минус матрицы с использованием constant пам€ти
template<class T> __global__ void __global__matrix_minus__constant__(T *b, int height, int width, size_t pitch) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			ELEMENT(T,b,i,j,pitch) = -ELEMENT(T,__c__,i,j,width*sizeof(T));
		}
	}
}
// ”нарный минус матрицы с использованием texture пам€ти
template<class T> __global__ void __global__matrix_minus__texture__(T *b, int height, int width, size_t pitch) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			uint4 buffer_a = tex1Dfetch(tex_a,IDX(i,j,width));
			ELEMENT(T,b,i,j,pitch) = -*(T*)&buffer_a;
		}
	}
}

template<class T> __host__ void __host__matrix_minus(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MEMORY src, MEMORY dest, MEMORY cache) {
	T * d_a;
	T * d_b;
	size_t pitch1;
	size_t pitch2;
	cudaError_t err;

	int height = a->height;
	int width =  a->width;
	b->height = height;
	b->width = width;

	switch(src) {
	case TEXTURE:
		err = cudaMallocHost((void**)&cpu_a, (size_t) a->width * a->height * sizeof(uint4));
		err = cudaMalloc((void**)&gpu_a, (size_t) a->width * a->height * sizeof(uint4));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				uint4 buffer;
				*(T*)&buffer = a->values[IDX(i,j,width)];
				cpu_a[IDX(i,j,width)] = buffer;
			}
		}
		//  настройка параемтров текстуры  texture
		tex_a.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_a.addressMode[1] = cudaAddressModeWrap;
		tex_a.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_a.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaBindTexture(0, tex_a, gpu_a, (size_t) a->width * a->height * sizeof(uint4));
		err = cudaMemcpy((void*)gpu_a, (void*)cpu_a, (size_t) a->width * a->height * sizeof(uint4), cudaMemcpyHostToDevice);
		err = cudaFreeHost((void*)cpu_a);
		break;
	case GLOBAL:
		err = cudaMallocPitch((void**)&d_a, &pitch1, (size_t) a->width * sizeof(T), (size_t) a->height);
		err = cudaMemcpy2D((void*)d_a, pitch1, (void*)a->values, (size_t) a->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		break;
	case CONSTANT:
		err = cudaMemcpyToSymbol(__c__, (void*)a->values, (size_t) a->height * a->width * sizeof(T), (size_t) 0, cudaMemcpyHostToDevice);
		break;
	default:
		break;
	}
	switch(dest) {
	case GLOBAL:
		err = cudaMallocPitch((void**)&d_b, &pitch2, (size_t) b->width * sizeof(T), (size_t) b->height);
		break;
	default:
		break;
	}

	if (src == GLOBAL && dest == GLOBAL) {
		__global__matrix_minus__global__<T><<<blocks,threads>>>(d_a, d_b, height, width, pitch1, pitch2);
	}
	else if (src == CONSTANT && dest == GLOBAL) {
		__global__matrix_minus__constant__<T><<<blocks,threads>>>(d_b, height, width, pitch2);
	}
	else if (src == TEXTURE && dest == GLOBAL) {
		__global__matrix_minus__texture__<T><<<blocks,threads>>>(d_b, height, width, pitch2);
	}

	switch(src) {
	case TEXTURE:
		err = cudaUnbindTexture(tex_a);
		err = cudaFree(gpu_a);
		break;
	case CONSTANT:
		break;
	default:
		err = cudaFree(d_a);
		break;
	}
	switch(dest) {
	case GLOBAL:
		err = cudaMemcpy2D((void*)b->values, (size_t) b->width * sizeof(T), (void*)d_b, pitch2, (size_t) width * sizeof(T), (size_t) height, cudaMemcpyDeviceToHost);
		err = cudaFree(d_b);
		break;
	default:
		break;
	}
}
