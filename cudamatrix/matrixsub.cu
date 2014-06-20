#include "matrix.h"
#include "globals.h"

// ¬ычитание двух матриц с использованием global пам€ти
template<class T> __global__ void __global__matrix_sub__global__(T *a, T *b, T *c, int height, int width, size_t pitch1, size_t pitch2, size_t pitch3, MEMORY cache) {
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
				ELEMENT(T,c,i,j,pitch3) = ELEMENT(T,a,i,j,pitch1) - ELEMENT(T,b,i,j,pitch2);
			}
		}
		break;
	case SHARED:
		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				__s__[IDX(i,j,sharedCountY)] = ELEMENT(T,a,sharedIndexX+i,sharedIndexY+j,pitch1);
				__s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1] = ELEMENT(T,b,sharedIndexX+i,sharedIndexY+j,pitch2);
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				ELEMENT(T,c,sharedIndexX+i,sharedIndexY+j,pitch3) = __s__[IDX(i,j,sharedCountY)] - __s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1];
			}
		}
		break;
	case LOCAL:
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				__l__[IDX(i,j,localCountY)] = ELEMENT(T,a,localIndexX+i,localIndexY+j,pitch1);
				__l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1] = ELEMENT(T,b,localIndexX+i,localIndexY+j,pitch2);
			}
		}
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				ELEMENT(T,c,localIndexX+i,localIndexY+j,pitch3) = __l__[IDX(i,j,localCountY)] - __l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1];
			}
		}
		break;
	}
}

// ¬ычитание двух матриц с использованием constant пам€ти
template<class T> __global__ void __global__matrix_sub__constant__(T *c, int height, int width, size_t pitch, MEMORY cache) {
	storage<T> buffer_a;
	storage<T> buffer_b;
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
				buffer_b.i = __c__[IDX(height+i,j,width)];
				ELEMENT(T,c,i,j,pitch) = buffer_a.t - buffer_b.t;
			}
		}
		break;
	case SHARED:
		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				buffer_a.i = __c__[IDX(sharedIndexX+i,sharedIndexY+j,width)];
				buffer_b.i = __c__[IDX(height+sharedIndexX+i,sharedIndexY+j,width)];
				__s__[IDX(i,j,sharedCountY)] = buffer_a.t;
				__s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1] = buffer_b.t;
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				ELEMENT(T,c,sharedIndexX+i,sharedIndexY+j,pitch) = __s__[IDX(i,j,sharedCountY)] - __s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1];
			}
		}
		break;
	case LOCAL:
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				buffer_a.i = __c__[IDX(localIndexX+i,localIndexY+j,width)];
				buffer_b.i = __c__[IDX(height+localIndexX+i,localIndexY+j,width)];
				__l__[IDX(i,j,localCountY)] = buffer_a.t;
				__l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1] = buffer_b.t;
			}
		}
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				ELEMENT(T,c,localIndexX+i,localIndexY+j,pitch) = __l__[IDX(i,j,localCountY)] - __l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1];
			}
		}
		break;
	}
}

// ¬ычитание двух матриц с использованием texture пам€ти
template<class T> __global__ void __global__matrix_sub__texture__(T *c, int height, int width, size_t pitch, MEMORY cache) {
	texturestorage<T> buffer_a;
	texturestorage<T> buffer_b;
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
				buffer_b.i = tex1Dfetch(tex_b,IDX(i,j,width));
				ELEMENT(T,c,i,j,pitch) = buffer_a.t - buffer_b.t;
			}
		}
		break;
	case SHARED:
		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				buffer_a.i = tex1Dfetch(tex_a,IDX(sharedIndexX+i,sharedIndexY+j,width));
				buffer_b.i = tex1Dfetch(tex_b,IDX(sharedIndexX+i,sharedIndexY+j,width));
				__s__[IDX(i,j,sharedCountY)] = buffer_a.t;
				__s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1] = buffer_b.t;
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
			for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
				ELEMENT(T,c,sharedIndexX+i,sharedIndexY+j,pitch) = __s__[IDX(i,j,sharedCountY)] - __s__[SHARED_SIZE - IDX(i,j,sharedCountY) - 1];
			}
		}
		break;
	case LOCAL:
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				buffer_a.i = tex1Dfetch(tex_a,IDX(localIndexX+i,localIndexY+j,width));
				buffer_b.i = tex1Dfetch(tex_b,IDX(localIndexX+i,localIndexY+j,width));
				__l__[IDX(i,j,localCountY)] = buffer_a.t;
				__l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1] = buffer_b.t;
			}
		}
		for (int i = 0; i < localCountX; i ++) {
			for (int j = 0; j < localCountY; j++) {
				ELEMENT(T,c,localIndexX+i,localIndexY+j,pitch) = __l__[IDX(i,j,localCountY)] - __l__[LOCAL_SIZE - IDX(i,j,localCountY) - 1];
			}
		}
		break;
	}
}

template<class T> __host__ void __cdecl __host__matrix_sub(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MATRIX<T> *c, MEMORY src, MEMORY dest, MEMORY cache) {
	T * h_a;
	T * h_b;
	T * d_a;
	T * d_b;
	T * d_c;
	size_t pitch1;
	size_t pitch2;
	size_t pitch3;
	cudaError_t err;
	storage<T> buffer;
	texturestorage<T> texturebuffer;

	int height = a->height;
	int width =  b->width;
	c->height = height;
	c->width = width;

	switch(src) {
	case TEXTURE:
		err = cudaMallocHost((void**)&cpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE));
		err = cudaMallocHost((void**)&cpu_b, (size_t) b->width * b->height * sizeof(TEXTURESTORAGETYPE));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				texturebuffer.t = a->values[IDX(i,j,width)];
				cpu_a[IDX(i,j,width)] = texturebuffer.i;
			}
		}
		for(int i=0; i<b->height; i++) {
			for(int j=0; j<b->width; j++) {
				texturebuffer.t = b->values[IDX(i,j,width)];
				cpu_b[IDX(i,j,width)] = texturebuffer.i;
			}
		}
		err = cudaMalloc((void**)&gpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE));
		err = cudaMalloc((void**)&gpu_b, (size_t) b->width * b->height * sizeof(TEXTURESTORAGETYPE));
		//  настройка параемтров текстуры  texture
		tex_a.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_a.addressMode[1] = cudaAddressModeWrap;
		tex_a.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_a.normalized     = false;                // не использовать нормализованную адресацию
		tex_b.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_b.addressMode[1] = cudaAddressModeWrap;
		tex_b.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_b.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaMemcpy((void*)gpu_a, (void*)cpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE), cudaMemcpyHostToDevice);
		err = cudaMemcpy((void*)gpu_b, (void*)cpu_b, (size_t) b->width * b->height * sizeof(TEXTURESTORAGETYPE), cudaMemcpyHostToDevice);
		err = cudaBindTexture(0, tex_a, gpu_a, (size_t) a->width * a->height * sizeof(TEXTURESTORAGETYPE));
		err = cudaBindTexture(0, tex_b, gpu_b, (size_t) b->width * b->height * sizeof(TEXTURESTORAGETYPE));
		err = cudaFreeHost((void*)cpu_a);
		err = cudaFreeHost((void*)cpu_b);
		break;
	case GLOBAL:
		err = cudaMallocPitch((void**)&d_a, &pitch1, (size_t) a->width * sizeof(T), a->height);
		err = cudaMallocPitch((void**)&d_b, &pitch2, (size_t) b->width * sizeof(T), b->height);
		err = cudaMemcpy2D((void*)d_a, pitch1, (void*)a->values, (size_t) a->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		err = cudaMemcpy2D((void*)d_b, pitch2, (void*)b->values, (size_t) b->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		break;
	case CONSTANT:
		err = cudaMallocHost((void**)&h_a, (size_t) a->width * a->height * sizeof(STORAGETYPE));
		err = cudaMallocHost((void**)&h_b, (size_t) b->width * b->height * sizeof(STORAGETYPE));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				buffer.t = a->values[IDX(i,j,width)];
				h_a[IDX(i,j,width)] = buffer.i;
			}
		}
		for(int i=0; i<b->height; i++) {
			for(int j=0; j<b->width; j++) {
				buffer.t = b->values[IDX(i,j,width)];
				h_b[IDX(i,j,width)] = buffer.i;
			}
		}
		err = cudaGetSymbolAddress((void **)&d_a, __c__);
		d_b=&d_a[height*width];
		err = cudaMemcpy(d_a, h_a, (size_t) height * width * sizeof(STORAGETYPE), cudaMemcpyHostToDevice);		
		err = cudaMemcpy(d_b, h_b, (size_t) height * width * sizeof(STORAGETYPE), cudaMemcpyHostToDevice);		
		err = cudaFreeHost((void*)h_a);
		err = cudaFreeHost((void*)h_b);
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
		__global__matrix_sub__global__<T><<<blocks,threads>>>(d_a, d_b, d_c, height, width, pitch1, pitch2, pitch3, cache);
	}
	else if (src == CONSTANT && dest == GLOBAL) {
		__global__matrix_sub__constant__<T><<<blocks,threads>>>(d_c, height, width, pitch3, cache);
	}
	else if (src == TEXTURE && dest == GLOBAL) {
		__global__matrix_sub__texture__<T><<<blocks,threads>>>(d_c, height, width, pitch3, cache);
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

template __host__ void __cdecl __host__matrix_sub<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, MATRIX<DATATYPE> *c, MEMORY src, MEMORY dest, MEMORY cache);
