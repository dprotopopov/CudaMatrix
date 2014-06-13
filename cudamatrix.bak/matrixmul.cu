#include "matrix.h"
#include "externs.h"

// Умножение двух матриц с использованием global памяти
template<class T> __global__ void __global__matrix_mul__global__none__(T *a, T *b, T *c, int height, int width_height, int width, size_t pitch1, size_t pitch2, size_t pitch3) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			T sum = 0;
			for (int k = 0; k < width_height ; k++ ) {
				sum += ELEMENT(T,a,i,k,pitch1) * ELEMENT(T,b,k,j,pitch2);
			}
			ELEMENT(T,c,i,j,pitch3) = sum;
		}
	}
}

// Умножение двух матриц с использованием shared памяти
// Using shared memory
// • Similar to local memory:
// – __shared__ float current_row[];
// • Only declare one variable as shared!
// – Multiple declarations of __shared__
// variables will occupy same memory space!
// – __shared__ float a[];
// – __shared__ float b[];
// – b[0] = 0.5f;
// – now a[0] == 0.5f also!
template<class T> __global__ void __global__matrix_mul__global__shared__(T *a, T *b, T *c, int height,int width_height, int width, size_t pitch1, size_t pitch2, size_t pitch3) {
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);
	
	__shared__ T __s__[SHARED_SIZE];
	
	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < width_height; j+= blockDim.y) {
			__s__[IDX(i,j,width_height)] = ELEMENT(T,a,sharedIndexX+i,j,pitch1);
		}
	}

	for (int i = threadIdx.x; i < width_height; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			__s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1] = ELEMENT(T,b,i,sharedIndexY+j,pitch2);
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			T sum = 0;
			for(int k=0; k<width_height; k++) {
				sum += __s__[IDX(i,k,width_height)] * __s__[SHARED_SIZE - IDX(k,j,sharedCountY) - 1];
			}
			ELEMENT(T,c,sharedIndexX+i,sharedIndexY+j,pitch3) = sum;
		}
	}
}

// Умножение двух матриц с использованием local памяти
// Local memory
// • Also read/write per-thread
// • Can’t read other threads’ local memory
// – Declare a variable in local memory using
// the __local__ keyword
// – __local__ float results[32];
// • Can index (this is where local arrays go)
// • Much slower than register memory!
// – Don’t use local arrays if you don’t have to
template<class T> __global__ void __global__matrix_mul__global__local__(T *a, T *b, T *c, int height,int width_height, int width, size_t pitch1, size_t pitch2, size_t pitch3) {
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/blockDim.x/gridDim.x);
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/blockDim.y/gridDim.y);
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/blockDim.x/gridDim.x)-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/blockDim.x/gridDim.x);
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/blockDim.y/gridDim.y)-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/blockDim.y/gridDim.y);
	
	T __l__[LOCAL_SIZE];
	
	for (int i = 0; i < localCountX; i ++) {
		for (int j = 0; j < width_height; j++) {
			__l__[IDX(i,j,width_height)] = ELEMENT(T,a,localIndexX+i,j,pitch1);
		}
	}

	for (int i = 0; i < width_height; i ++) {
		for (int j = 0; j < localCountY; j++) {
			__l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1] = ELEMENT(T,b,i,localIndexY+j,pitch2);
		}
	}

	for (int i = 0; i < localCountX; i++ ) {
		for (int j = 0; j < localCountY; j++ ) {
			T sum = 0;
			for(int k=0; k<width_height; k++) {
				sum += __l__[IDX(i,k,width_height)] * __l__[LOCAL_SIZE - IDX(k,j,localCountY) - 1];
			}
			ELEMENT(T,c,localIndexX+i,localIndexY+j,pitch3) = sum;
		}
	}
}

// Умножение двух матриц с использованием global памяти
template<class T> __global__ void __global__matrix_mul__constant__none__(T *c, int height, int width_height, int width, size_t pitch) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			T sum = 0;
			for (int k = 0; k < width_height ; k++ ) {
				sum += ELEMENT(T,__c__,i,k,width_height*sizeof(T)) * ELEMENT(T,__c__,k,j+IDX(height,0,width_height),width*sizeof(T));
			}
			ELEMENT(T,c,i,j,pitch) = sum;
		}
	}
}

// Умножение двух матриц с использованием shared памяти
// Using shared memory
// • Similar to local memory:
// – __shared__ float current_row[];
// • Only declare one variable as shared!
// – Multiple declarations of __shared__
// variables will occupy same memory space!
// – __shared__ float a[];
// – __shared__ float b[];
// – b[0] = 0.5f;
// – now a[0] == 0.5f also!
template<class T> __global__ void __global__matrix_mul__constant__shared__(T *c, int height,int width_height, int width, size_t pitch) {
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);
	
	__shared__ T __s__[SHARED_SIZE];
	
	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < width_height; j+= blockDim.y) {
			__s__[IDX(i,j,width_height)] = ELEMENT(T,__c__,sharedIndexX+i,j,width_height*sizeof(T));
		}
	}

	for (int i = threadIdx.x; i < width_height; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			__s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1] = ELEMENT(T,__c__,i,sharedIndexY+j+IDX(height,0,width_height),width*sizeof(T));
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			T sum = 0;
			for(int k=0; k<width_height; k++) {
				sum += __s__[IDX(i,k,width_height)] * __s__[SHARED_SIZE - IDX(k,j,sharedCountY) - 1];
			}
			ELEMENT(T,c,sharedIndexX+i,sharedIndexY+j,pitch) = sum;
		}
	}
}

// Умножение двух матриц с использованием local памяти
// Local memory
// • Also read/write per-thread
// • Can’t read other threads’ local memory
// – Declare a variable in local memory using
// the __local__ keyword
// – __local__ float results[32];
// • Can index (this is where local arrays go)
// • Much slower than register memory!
// – Don’t use local arrays if you don’t have to
template<class T> __global__ void __global__matrix_mul__constant__local__(T *c, int height,int width_height, int width, size_t pitch) {
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/blockDim.x/gridDim.x);
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/blockDim.y/gridDim.y);
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/blockDim.x/gridDim.x)-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/blockDim.x/gridDim.x);
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/blockDim.y/gridDim.y)-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/blockDim.y/gridDim.y);
	
	T __l__[LOCAL_SIZE];
	
	for (int i = 0; i < localCountX; i ++) {
		for (int j = 0; j < width_height; j++) {
			__l__[IDX(i,j,width_height)] = ELEMENT(T,__c__,localIndexX+i,j, width_height*sizeof(T));
		}
	}

	for (int i = 0; i < width_height; i ++) {
		for (int j = 0; j < localCountY; j++) {
			__l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1] = ELEMENT(T,__c__,i,localIndexY+j+IDX(height,0,width_height), width*sizeof(T));
		}
	}

	for (int i = 0; i < localCountX; i++ ) {
		for (int j = 0; j < localCountY; j++ ) {
			T sum = 0;
			for(int k=0; k<width_height; k++) {
				sum += __l__[IDX(i,k,width_height)] * __l__[LOCAL_SIZE - IDX(k,j,localCountY) - 1];
			}
			ELEMENT(T,c,localIndexX+i,localIndexY+j,pitch) = sum;
		}
	}
}

// Умножение двух матриц с использованием texture памяти
template<class T> __global__ void __global__matrix_mul__texture__none__(T *c, int height, int width_height, int width, size_t pitch) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			T sum = 0;
			for (int k = 0; k < width_height ; k++ ) {
				uint4 buffer_a = tex1Dfetch(tex_a,IDX(i,k,width_height));
				uint4 buffer_b = tex1Dfetch(tex_b,IDX(k,j,width));
				sum += *(T*)&buffer_a + *(T*)&buffer_b;
			}
			ELEMENT(T,c,i,j,pitch) = sum;
		}
	}
}

// Умножение двух матриц с использованием shared памяти
// Using shared memory
// • Similar to local memory:
// – __shared__ float current_row[];
// • Only declare one variable as shared!
// – Multiple declarations of __shared__
// variables will occupy same memory space!
// – __shared__ float a[];
// – __shared__ float b[];
// – b[0] = 0.5f;
// – now a[0] == 0.5f also!
template<class T> __global__ void __global__matrix_mul__texture__shared__(T *c, int height,int width_height, int width, size_t pitch) {
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);
	
	__shared__ T __s__[SHARED_SIZE];
	
	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < width_height; j+= blockDim.y) {
			uint4 buffer_a = tex1Dfetch(tex_a,IDX(sharedIndexX+i,j,width_height));
			__s__[IDX(i,j,width_height)] = *(T*)&buffer_a;
		}
	}

	for (int i = threadIdx.x; i < width_height; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			uint4 buffer_b = tex1Dfetch(tex_b,IDX(i,sharedIndexY+j,width));
			__s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1] = *(T*)&buffer_b;
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			T sum = 0;
			for(int k=0; k<width_height; k++) {
				sum += __s__[IDX(i,k,width_height)] * __s__[SHARED_SIZE - IDX(k,j,sharedCountY) - 1];
			}
			ELEMENT(T,c,sharedIndexX+i,sharedIndexY+j,pitch) = sum;
		}
	}
}

// Умножение двух матриц с использованием local памяти
// Local memory
// • Also read/write per-thread
// • Can’t read other threads’ local memory
// – Declare a variable in local memory using
// the __local__ keyword
// – __local__ float results[32];
// • Can index (this is where local arrays go)
// • Much slower than register memory!
// – Don’t use local arrays if you don’t have to
template<class T> __global__ void __global__matrix_mul__texture__local__(T *c, int height,int width_height, int width, size_t pitch) {
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/blockDim.x/gridDim.x);
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/blockDim.y/gridDim.y);
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/blockDim.x/gridDim.x)-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/blockDim.x/gridDim.x);
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/blockDim.y/gridDim.y)-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/blockDim.y/gridDim.y);
	
	T __l__[LOCAL_SIZE];
	
	for (int i = 0; i < localCountX; i ++) {
		for (int j = 0; j < width_height; j++) {
			uint4 buffer_a = tex1Dfetch(tex_a,IDX(localIndexX+i,j,width_height));
			__l__[IDX(i,j,width_height)] = *(T*)&buffer_a;
		}
	}

	for (int i = 0; i < width_height; i ++) {
		for (int j = 0; j < localCountY; j++) {
			uint4 buffer_b = tex1Dfetch(tex_b,IDX(i,localIndexY+j,width));
			__l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1] = *(T*)&buffer_b;
		}
	}

	for (int i = 0; i < localCountX; i++ ) {
		for (int j = 0; j < localCountY; j++ ) {
			T sum = 0;
			for(int k=0; k<width_height; k++) {
				sum += __l__[IDX(i,k,width_height)] * __l__[LOCAL_SIZE - IDX(k,j,localCountY) - 1];
			}
			ELEMENT(T,c,localIndexX+i,localIndexY+j,pitch) = sum;
		}
	}
}

template<class T> __host__ void __host__matrix_mul(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MATRIX<T> *c, MEMORY src, MEMORY dest, MEMORY cache) {
	T * d_a;
	T * d_b;
	T * d_c;
	size_t pitch1;
	size_t pitch2;
	size_t pitch3;
	cudaError_t err;

	int height = a->height;
	int width_height = a->width;
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
		err = cudaMemcpyToSymbol(__c__, (void*)b->values, (size_t) b->height * b->width * sizeof(T), (size_t) height*width_height*sizeof(T), cudaMemcpyHostToDevice);
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

	if (src == GLOBAL && dest == GLOBAL && cache == NONE) {
		__global__matrix_mul__global__none__<T><<<blocks,threads>>>(d_a, d_b, d_c, height, width_height, width, pitch1, pitch2, pitch3);
	}
	else if (src == GLOBAL && dest == GLOBAL && cache == LOCAL) {
		__global__matrix_mul__global__local__<T><<<blocks,threads>>>(d_a, d_b, d_c, height, width_height, width, pitch1, pitch2, pitch3);
	}
	else if (src == GLOBAL && dest == GLOBAL && cache == SHARED) {
		__global__matrix_mul__global__shared__<T><<<blocks,threads>>>(d_a, d_b, d_c, height, width_height, width, pitch1, pitch2, pitch3);
	}
	else if (src == CONSTANT && dest == GLOBAL && cache == NONE) {
		__global__matrix_mul__constant__none__<T><<<blocks,threads>>>(d_c, height, width_height, width, pitch3);
	}
	else if (src == CONSTANT && dest == GLOBAL && cache == LOCAL) {
		__global__matrix_mul__constant__local__<T><<<blocks,threads>>>(d_c, height, width_height, width, pitch3);
	}
	else if (src == CONSTANT && dest == GLOBAL && cache == SHARED) {
		__global__matrix_mul__constant__shared__<T><<<blocks,threads>>>(d_c, height, width_height, width, pitch3);
	}
	else if (src == TEXTURE && dest == GLOBAL && cache == NONE) {
		__global__matrix_mul__texture__none__<T><<<blocks,threads>>>(d_c, height, width_height, width, pitch3);
	}
	else if (src == TEXTURE && dest == GLOBAL && cache == LOCAL) {
		__global__matrix_mul__texture__local__<T><<<blocks,threads>>>(d_c, height, width_height, width, pitch3);
	}
	else if (src == TEXTURE && dest == GLOBAL && cache == SHARED) {
		__global__matrix_mul__texture__shared__<T><<<blocks,threads>>>(d_c, height, width_height, width, pitch3);
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


