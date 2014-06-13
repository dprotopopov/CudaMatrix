#include "matrix.h"
#include "globals.h"

// ”нарный плюс матрицы с использованием global пам€ти
template<class T> __global__ void __global__matrix_plus__global__(T *a, T *b, int height, int width, size_t pitch1, size_t pitch2) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			ELEMENT(T,b,i,j,pitch2) = ELEMENT(T,a,i,j,pitch1);
		}
	}
}

// ”нарный плюс матрицы с использованием constant пам€ти
template<class T> __global__ void __global__matrix_plus__constant__(T *b, int height, int width, size_t pitch) {
	storage<T> buffer_a;
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			buffer_a.i = __c__[IDX(i,j,width)];
			ELEMENT(T,b,i,j,pitch) = buffer_a.t;
		}
	}
}

// ”нарный плюс матрицы с использованием texture пам€ти
template<class T> __global__ void __global__matrix_plus__texture__(T *b, int height, int width, size_t pitch) {
	storage<T> buffer_a;
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			buffer_a.i = tex1Dfetch(tex_a,IDX(i,j,width));
			ELEMENT(T,b,i,j,pitch) = buffer_a.t;
		}
	}
}

template<class T> __host__ void __cdecl __host__matrix_plus(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MEMORY src, MEMORY dest, MEMORY cache) {

	T * d_a;
	T * d_b;
	size_t pitch1;
	size_t pitch2;
	cudaError_t err;
	storage<T> buffer;

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
				buffer.t = a->values[IDX(i,j,width)];
				cpu_a[IDX(i,j,width)] = buffer.i;
			}
		}
		//  настройка параемтров текстуры  texture
		tex_a.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_a.addressMode[1] = cudaAddressModeWrap;
		tex_a.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_a.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaMemcpy((void*)gpu_a, (void*)cpu_a, (size_t) a->width * a->height * sizeof(uint4), cudaMemcpyHostToDevice);
		err = cudaBindTexture(0, tex_a, gpu_a, (size_t) a->width * a->height * sizeof(uint4));
		err = cudaFreeHost((void*)cpu_a);
		break;
	case GLOBAL:
		err = cudaMallocPitch((void**)&d_a, &pitch1, (size_t) a->width * sizeof(T), (size_t) a->height);
		err = cudaMemcpy2D((void*)d_a, pitch1, (void*)a->values, (size_t) a->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		break;
	case CONSTANT:
		err = cudaMallocHost((void**)&cpu_a, (size_t) a->width * a->height * sizeof(uint4));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				buffer.t = a->values[IDX(i,j,width)];
				cpu_a[IDX(i,j,width)] = buffer.i;
			}
		}
		err = cudaGetSymbolAddress((void **)&gpu_a, __c__);
		err = cudaMemcpy(gpu_a, cpu_a, (size_t) height * width * sizeof(uint4), cudaMemcpyHostToDevice);		
		err = cudaFreeHost((void*)cpu_a);
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
		__global__matrix_plus__global__<T><<<blocks,threads>>>(d_a, d_b, height, width, pitch1, pitch2);
	}
	else if (src == CONSTANT && dest == GLOBAL) {
		__global__matrix_plus__constant__<T><<<blocks,threads>>>(d_b, height, width, pitch2);
	}
	else if (src == TEXTURE && dest == GLOBAL) {
		__global__matrix_plus__texture__<T><<<blocks,threads>>>(d_b, height, width, pitch2);
	}

	switch(src) {
	case TEXTURE:
		err = cudaUnbindTexture(tex_a);
		err = cudaFree(gpu_a);
		break;
	case CONSTANT:
		break;
	default:
		cudaFree(d_a);
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
	
	err = err;
}

template __host__ void __cdecl __host__matrix_plus<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MEMORY src, MEMORY dest, MEMORY cache);
