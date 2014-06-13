#include "matrix.h"
#include "externs.h"

// ¬ычитание двух матриц с использованием global пам€ти
template<class T> __global__ void __global__matrix_sub__global__(T *a, T *b, T *c, int height, int width, size_t pitch1, size_t pitch2, size_t pitch3) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			ELEMENT(T,c,i,j,pitch3) = ELEMENT(T,a,i,j,pitch1) - ELEMENT(T,b,i,j,pitch2);
		}
	}
}

// ¬ычитание двух матриц с использованием constant пам€ти
template<class T> __global__ void __global__matrix_sub__constant__(T *c, int height, int width, size_t pitch) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			ELEMENT(T,c,i,j,pitch) = ELEMENT(T,__c__,i,j,width*sizeof(T)) - ELEMENT(T,__c__,height+i,j,width*sizeof(T));
		}
	}
}

// ¬ычитание двух матриц с использованием texture пам€ти
template<class T> __global__ void __global__matrix_sub__texture__(T *c, int height, int width, size_t pitch) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			uint4 buffer_a = tex1Dfetch(tex_a,IDX(i,j,width));
			uint4 buffer_b = tex1Dfetch(tex_b,IDX(i,j,width));
			ELEMENT(T,c,i,j,pitch) = *(T*)&buffer_a - *(T*)&buffer_b;
		}
	}
}

template<class T> __host__ void __host__matrix_sub(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MATRIX<T> *c, MEMORY src, MEMORY dest, MEMORY cache) {
	T * d_a;
	T * d_b;
	T * d_c;
	size_t pitch1;
	size_t pitch2;
	size_t pitch3;
	cudaError_t err;

	int height = a->height;
	int width =  b->width;
	c->height = height;
	c->width = width;

	switch(src) {
	case TEXTURE:
		err = cudaMallocHost((void**)&cpu_a, (size_t) a->width * a->height * sizeof(uint4));
		err = cudaMallocHost((void**)&cpu_b, (size_t) b->width * b->height * sizeof(uint4));
		err = cudaMalloc((void**)&gpu_a, (size_t) a->width * a->height * sizeof(uint4));
		err = cudaMalloc((void**)&gpu_b, (size_t) b->width * b->height * sizeof(uint4));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				uint4 buffer;
				*(T*)&buffer = a->values[IDX(i,j,width)];
				cpu_a[IDX(i,j,width)] = buffer;
			}
		}
		for(int i=0; i<b->height; i++) {
			for(int j=0; j<b->width; j++) {
				uint4 buffer;
				*(T*)&buffer = b->values[IDX(i,j,width)];
				cpu_b[IDX(i,j,width)] = buffer;
			}
		}
		//  настройка параемтров текстуры  texture
		tex_a.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_a.addressMode[1] = cudaAddressModeWrap;
		tex_a.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_a.normalized     = false;                // не использовать нормализованную адресацию
		tex_b.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_b.addressMode[1] = cudaAddressModeWrap;
		tex_b.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_b.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaBindTexture(0, tex_a, gpu_a, (size_t) a->width * a->height * sizeof(uint4));
		err = cudaBindTexture(0, tex_b, gpu_b, (size_t) b->width * b->height * sizeof(uint4));
		err = cudaMemcpy((void*)gpu_a, (void*)cpu_a, (size_t) a->width * a->height * sizeof(uint4), cudaMemcpyHostToDevice);
		err = cudaMemcpy((void*)gpu_b, (void*)cpu_b, (size_t) b->width * b->height * sizeof(uint4), cudaMemcpyHostToDevice);
		err = cudaFreeHost((void*)cpu_a);
		err = cudaFreeHost((void*)cpu_b);
		break;
	case GLOBAL:
		err = cudaMallocPitch((void**)&d_a, &pitch1, (size_t) a->width * sizeof(T), (size_t) a->height);
		err = cudaMallocPitch((void**)&d_b, &pitch2, (size_t) b->width * sizeof(T), (size_t) b->height);
		err = cudaMemcpy2D((void*)d_a, pitch1, (void*)a->values, (size_t) a->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		err = cudaMemcpy2D((void*)d_b, pitch2, (void*)b->values, (size_t) b->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		break;
	case CONSTANT:
		err = cudaMemcpyToSymbol(__c__, (void*)a->values, (size_t) a->height * a->width * sizeof(T), (size_t) 0, cudaMemcpyHostToDevice);
		err = cudaMemcpyToSymbol(__c__, (void*)b->values, (size_t) b->height * b->width * sizeof(T), (size_t) height*width*sizeof(T), cudaMemcpyHostToDevice);
		break;
	default:
		break;
	}
	switch(dest) {
	case GLOBAL:
		err = cudaMallocPitch((void**)&d_c, &pitch3, (size_t) c->width * sizeof(T), (size_t) c->height);
		break;
	default:
		break;
	}

	if (src == GLOBAL && dest == GLOBAL) {
		__global__matrix_sub__global__<T><<<blocks,threads>>>(d_a, d_b, d_c, height, width, pitch1, pitch2, pitch3);
	}
	else if (src == CONSTANT && dest == GLOBAL) {
		__global__matrix_sub__constant__<T><<<blocks,threads>>>(d_c, height, width, pitch3);
	}
	else if (src == TEXTURE && dest == GLOBAL) {
		__global__matrix_sub__texture__<T><<<blocks,threads>>>(d_c, height, width, pitch3);
	}

	switch(src) {
	case TEXTURE:
		err = cudaUnbindTexture(tex_a);
		err = cudaUnbindTexture(tex_b);
		err = cudaFree(gpu_a);
		err = cudaFree(gpu_b);
		break;
	case CONSTANT:
		break;
	default:
		err = cudaFree(d_a);
		err = cudaFree(d_b);
		break;
	}
	switch(dest) {
	case GLOBAL:
		err = cudaMemcpy2D((void*)c->values, (size_t) c->width * sizeof(T), (void*)d_c, pitch3, (size_t) width * sizeof(T), (size_t) height, cudaMemcpyDeviceToHost);
		err = cudaFree(d_c);
		break;
	default:
		break;
	}
	
	err = err;
}
