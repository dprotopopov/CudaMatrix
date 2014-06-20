#include "matrix.h"
#include "globals.h"

// Tранспонирование матрицы с использованием global памяти
template<class T> __global__ void __global__matrix_rot__global__(T *a, T *b, int height, int width, size_t pitch1, size_t pitch2, MEMORY cache) {
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/(blockDim.x*gridDim.x))-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/(blockDim.y*gridDim.y))-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));

	T __l__[LOCAL_SIZE];
	__shared__ T __s__[SHARED_SIZE];

	switch(cache){
	case NONE:
		for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
			for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
				ELEMENT(T,b,j,i,pitch2) = ELEMENT(T,a,i,j,pitch1);
			}
		}
		break;
	case SHARED:
		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				__s__[IDX(i,j,sharedCountY)] = ELEMENT(T,a,sharedIndexX+i,sharedIndexY+j,pitch1);
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				ELEMENT(T,b,sharedIndexY+j,sharedIndexX+i,pitch2) = __s__[IDX(i,j,sharedCountY)];
			}
		}
		break;
	case LOCAL:
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				__l__[IDX(i,j,localCountY)] = ELEMENT(T,a,localIndexX+i,localIndexY+j,pitch1);
			}
		}
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				ELEMENT(T,b,localIndexY+j,localIndexX+i,pitch2) = __l__[IDX(i,j,localCountY)];
			}
		}
		break;
	}
}

// Tранспонирование матрицы с использованием constant памяти
template<class T> __global__ void __global__matrix_rot__constant__(T *b, int height, int width, size_t pitch, MEMORY cache) {
	storage<T> buffer_a;
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/(blockDim.x*gridDim.x))-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/(blockDim.y*gridDim.y))-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));

	T __l__[LOCAL_SIZE];
	__shared__ T __s__[SHARED_SIZE];

	switch(cache){
	case NONE:
		for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
			for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
				buffer_a.i = __c__[IDX(i,j,width)];
				ELEMENT(T,b,j,i,pitch) = buffer_a.t;
			}
		}
		break;
	case SHARED:
		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				buffer_a.i = __c__[IDX(sharedIndexX+i,sharedIndexY+j,width)];
				__s__[IDX(i,j,sharedCountY)] = buffer_a.t;
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				ELEMENT(T,b,sharedIndexY+j,sharedIndexX+i,pitch) = __s__[IDX(i,j,sharedCountY)];
			}
		}
		break;
	case LOCAL:
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				buffer_a.i = __c__[IDX(localIndexX+i,localIndexY+j,width)];
				__l__[IDX(i,j,localCountY)] = buffer_a.t;
			}
		}
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				ELEMENT(T,b,localIndexY+j,localIndexX+i,pitch) = __l__[IDX(i,j,localCountY)];
			}
		}
		break;
	}
}

// Tранспонирование матрицы с использованием texture памяти
template<class T> __global__ void __global__matrix_rot__texture__(T *b, int height, int width, size_t pitch, MEMORY cache) {
	texturestorage<T> buffer_a;
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/(blockDim.x*gridDim.x))-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/(blockDim.y*gridDim.y))-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));

	T __l__[LOCAL_SIZE];
	__shared__ T __s__[SHARED_SIZE];

	switch(cache){
	case NONE:
		for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
			for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
				buffer_a.i = tex1Dfetch(tex_a,IDX(i,j,width));
				ELEMENT(T,b,j,i,pitch) = buffer_a.t;
			}
		}
		break;
	case SHARED:
		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				buffer_a.i = tex1Dfetch(tex_a,IDX(sharedIndexX+i,sharedIndexY+j,width));
				__s__[IDX(i,j,sharedCountY)] = buffer_a.t;
			}
		}
		__syncthreads();

		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				ELEMENT(T,b,sharedIndexY+j,sharedIndexX+i,pitch) = __s__[IDX(i,j,sharedCountY)];
			}
		}
		break;
	case LOCAL:
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				buffer_a.i = tex1Dfetch(tex_a,IDX(localIndexX+i,localIndexY+j,width));
				__l__[IDX(i,j,localCountY)] = buffer_a.t;
			}
		}
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				ELEMENT(T,b,localIndexY+j,localIndexX+i,pitch) = __l__[IDX(i,j,localCountY)];
			}
		}
		break;
	}
}

template<class T> __host__ void __cdecl __host__matrix_rot(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MEMORY src, MEMORY dest, MEMORY cache) {
	T * h_a;
	T * d_a;
	T * d_b;
	size_t pitch1;
	size_t pitch2;
	cudaError_t err;
	storage<T> buffer;
	texturestorage<T> texturebuffer;

	int height = a->height;
	int width =  a->width;
	b->height = width;
	b->width = height;

	switch(src) {
	case TEXTURE:
		err = cudaMallocHost((void**)&cpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE));
		err = cudaMalloc((void**)&gpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				texturebuffer.t = a->values[IDX(i,j,width)];
				cpu_a[IDX(i,j,width)] = texturebuffer.i;
			}
		}
		//  настройка параемтров текстуры  texture
		tex_a.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_a.addressMode[1] = cudaAddressModeWrap;
		tex_a.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_a.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaMemcpy((void*)gpu_a, (void*)cpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE), cudaMemcpyHostToDevice);
		err = cudaBindTexture(0, tex_a, gpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE));
		err = cudaFreeHost((void*)cpu_a);
		break;
	case GLOBAL:
		err = cudaMallocPitch((void**)&d_a, &pitch1, (size_t) a->width * sizeof(T), (size_t) a->height);
		err = cudaMemcpy2D((void*)d_a, pitch1, (void*)a->values, (size_t) a->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		break;
	case CONSTANT:
		err = cudaMallocHost((void**)&h_a, (size_t) a->width * a->height * sizeof(STORAGETYPE));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				buffer.t = a->values[IDX(i,j,width)];
				h_a[IDX(i,j,width)] = buffer.i;
			}
		}
		err = cudaGetSymbolAddress((void **)&d_a, __c__);
		err = cudaMemcpy(d_a, h_a, (size_t) height * width * sizeof(STORAGETYPE), cudaMemcpyHostToDevice);		
		err = cudaFreeHost((void*)h_a);
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
		__global__matrix_rot__global__<T><<<blocks,threads>>>(d_a, d_b, height, width, pitch1, pitch2, cache);
	}
	else if (src == CONSTANT && dest == GLOBAL) {
		__global__matrix_rot__constant__<T><<<blocks,threads>>>(d_b, height, width, pitch2, cache);
	}
	else if (src == TEXTURE && dest == GLOBAL) {
		__global__matrix_rot__texture__<T><<<blocks,threads>>>(d_b, height, width, pitch2, cache);
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
		err = cudaMemcpy2D((void*)b->values, (size_t) b->width * sizeof(T), (void*)d_b, pitch2, (size_t) height * sizeof(T), (size_t) width, cudaMemcpyDeviceToHost);
		err = cudaFree(d_b);
		break;
	default:
		break;
	}
	
	err = err;
}

template __host__ void __cdecl __host__matrix_rot<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, MEMORY src, MEMORY dest, MEMORY cache);
