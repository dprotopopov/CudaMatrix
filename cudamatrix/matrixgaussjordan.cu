#include "matrix.h"
#include "globals.h"

template<class T> __global__ void __global__matrix_gaussjordanstep__global__none__(T *prev, T *next, int row, int col, int height, int width, size_t pitch) {
	T d = ELEMENT(T,prev,row,col,pitch);
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			if (i!=row && j!=col) {
				ELEMENT(T,next,i,j,pitch) = ELEMENT(T,prev,i,j,pitch) - ELEMENT(T,prev,i,col,pitch)*ELEMENT(T,prev,row,j,pitch)/ d;
			}
			else if (i==row && j!=col) {
				ELEMENT(T,next,i,j,pitch) = ELEMENT(T,prev,i,j,pitch) / d;
			}
			else if (i!=row && j==col) {
				ELEMENT(T,next,i,j,pitch) = (T)0;
			}
			else {
				ELEMENT(T,next,i,j,pitch) = (T)1;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep2__global__none__(T *buffer, T *prev, T *next, int row, int col, int height, int width, size_t pitch) {
	T d = buffer[row];
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			if (i!=row) {
				ELEMENT(T,next,i,j,pitch) = ELEMENT(T,prev,i,j,pitch) - buffer[i]*ELEMENT(T,prev,row,j,pitch)/ d;
			}
			else {
				ELEMENT(T,next,i,j,pitch) = ELEMENT(T,prev,i,j,pitch) / d;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep__global__shared__(T *prev, T *next, int row, int col, int height, int width, size_t pitch) {
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);

	T d = ELEMENT(T,prev,row,col,pitch);

	__shared__ T __s__[SHARED_SIZE];

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < 1; j+= blockDim.y) {
			__s__[i] = ELEMENT(T,prev,sharedIndexX+i,col,pitch);
		}
	}
	for (int i = threadIdx.x; i < 1; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			__s__[SHARED_SIZE - j - 1] = ELEMENT(T,prev,row,sharedIndexY+j,pitch);
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j += blockDim.y) {
			if (sharedIndexX+i!=row && sharedIndexY+j!=col) {
				ELEMENT(T,next,sharedIndexX+i,sharedIndexY+j,pitch) = ELEMENT(T,prev,sharedIndexX+i,sharedIndexY+j,pitch) - __s__[i]*__s__[SHARED_SIZE - j - 1]/ d;
			}
			else if (sharedIndexX+i==row && sharedIndexY+j!=col) {
				ELEMENT(T,next,sharedIndexX+i,sharedIndexY+j,pitch) = __s__[SHARED_SIZE - j - 1] /d;
			}
			else if (sharedIndexX+i!=row && sharedIndexY+j==col) {
				ELEMENT(T,next,sharedIndexX+i,sharedIndexY+j,pitch) = (T)0;
			}
			else {
				ELEMENT(T,next,sharedIndexX+i,sharedIndexY+j,pitch) = (T)1;
			}
		}
	}
}
template<class T> __global__ void __global__matrix_gaussjordanstep2__global__shared__(T *buffer, T *prev, T *next, int row, int col, int height, int width, size_t pitch) {
	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);

	T d = buffer[row];

	__shared__ T __s__[SHARED_SIZE];

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < 1; j+= blockDim.y) {
			__s__[i] = buffer[sharedIndexX+i];
		}
	}
	for (int i = threadIdx.x; i < 1; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			__s__[SHARED_SIZE - j - 1] = ELEMENT(T,prev,row,sharedIndexY+j,pitch);
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j += blockDim.y) {
			if (sharedIndexX+i!=row) {
				ELEMENT(T,next,sharedIndexX+i,sharedIndexY+j,pitch) = ELEMENT(T,prev,sharedIndexX+i,sharedIndexY+j,pitch) - __s__[i]*__s__[SHARED_SIZE - j - 1]/ d;
			}
			else {
				ELEMENT(T,next,sharedIndexX+i,sharedIndexY+j,pitch) = __s__[SHARED_SIZE - j - 1] / d;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep__global__local__(T *prev, T *next, int row, int col, int height, int width, size_t pitch) {
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x))-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/(blockDim.x*gridDim.x));
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y))-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/(blockDim.y*gridDim.y));

	T d = ELEMENT(T,prev,row,col,pitch);

	T __l__[LOCAL_SIZE];

	for (int i = 0; i < localCountX; i++) {
		for (int j = 0; j < 1; j++) {
			__l__[i] = ELEMENT(T,prev,localIndexX+i,col,pitch);
		}
	}
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < localCountY; j++) {
			__l__[LOCAL_SIZE - j - 1] = ELEMENT(T,prev,row,localIndexY+j,pitch);
		}
	}

	for (int i = 0; i < localCountX; i++ ) {
		for (int j = 0; j < localCountY; j++ ) {
			if (localIndexX+i!=row && localIndexY+j!=col) {
				ELEMENT(T,next,localIndexX+i,localIndexY+j,pitch) = ELEMENT(T,prev,localIndexX+i,localIndexY+j,pitch) - __l__[i]*__l__[LOCAL_SIZE - j - 1]/ d;
			}
			else if (localIndexX+i==row && localIndexY+j!=col) {
				ELEMENT(T,next,localIndexX+i,localIndexY+j,pitch) = __l__[LOCAL_SIZE - j - 1] / d;
			}
			else if (localIndexX+i!=row && localIndexY+j==col) {
				ELEMENT(T,next,localIndexX+i,localIndexY+j,pitch) = (T)0;
			}
			else {
				ELEMENT(T,next,localIndexX+i,localIndexY+j,pitch) = (T)1;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep2__global__local__(T *buffer, T *prev, T *next, int row, int col, int height, int width, size_t pitch) {
	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x))-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/(blockDim.x*gridDim.x));
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y))-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/(blockDim.y*gridDim.y));

	T d = buffer[row];

	T __l__[LOCAL_SIZE];

	for (int i = 0; i < localCountX; i++) {
		for (int j = 0; j < 1; j++) {
			__l__[i] = buffer[localIndexX+i];
		}
	}
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < localCountY; j++) {
			__l__[LOCAL_SIZE - j - 1] = ELEMENT(T,prev,row,localIndexY+j,pitch);
		}
	}

	for (int i = 0; i < localCountX; i++ ) {
		for (int j = 0; j < localCountY; j++ ) {
			if (localIndexX+i!=row) {
				ELEMENT(T,next,localIndexX+i,localIndexY+j,pitch) = ELEMENT(T,prev,localIndexX+i,localIndexY+j,pitch) - __l__[i]*__l__[LOCAL_SIZE - j - 1]/ d;
			}
			else {
				ELEMENT(T,next,localIndexX+i,localIndexY+j,pitch) = __l__[LOCAL_SIZE - j - 1] / d;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep__texture__none__(uint4 *next, int row, int col, int height, int width) {
	storage<T> buffer_a;
	storage<T> buffer_b;
	storage<T> buffer_c;
	storage<T> buffer_d;
	storage<T> buffer_e;
	buffer_d.i = tex1Dfetch(tex_w,IDX(row,col,width));
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			if (i!=row && j!=col) {
				buffer_a.i = tex1Dfetch(tex_w,IDX(i,j,width));
				buffer_b.i = tex1Dfetch(tex_w,IDX(i,col,width));
				buffer_c.i = tex1Dfetch(tex_w,IDX(row,j,width));
				buffer_e.t =  buffer_a.t - buffer_b.t*buffer_c.t/ buffer_d.t;
				next[IDX(i,j,width)] = buffer_e.i;
			}
			else if (i==row && j!=col) {
				buffer_a.i = tex1Dfetch(tex_w,IDX(i,j,width));
				buffer_e.t = buffer_a.t / buffer_d.t;
				next[IDX(i,j,width)] = buffer_e.i;
			}
			else if (i!=row && j==col) {
				buffer_e.t = (T)0;
				next[IDX(i,j,width)] = buffer_e.i;
			}
			else {
				buffer_e.t = (T)1;
				next[IDX(i,j,width)] = buffer_e.i;
			}
		}
	}
}
template<class T> __global__ void __global__matrix_gaussjordanstep2__texture__none__(uint4 *next, int row, int col, int height, int width) {
	storage<T> buffer_a;
	storage<T> buffer_b;
	storage<T> buffer_c;
	storage<T> buffer_d;
	storage<T> buffer_e;
	buffer_d.i = tex1Dfetch(tex_buffer,row);
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		for (int j = blockDim.y*blockIdx.y + threadIdx.y; j < width; j += blockDim.y*gridDim.y) {
			if (i!=row) {
				buffer_a.i = tex1Dfetch(tex_w,IDX(i,j,width));
				buffer_b.i = tex1Dfetch(tex_buffer,i);
				buffer_c.i = tex1Dfetch(tex_w,IDX(row,j,width));
				buffer_e.t =  buffer_a.t - buffer_b.t*buffer_c.t/ buffer_d.t;
				next[IDX(i,j,width)] = buffer_e.i;
			}
			else {
				buffer_a.i = tex1Dfetch(tex_w,IDX(i,j,width));
				buffer_e.t = buffer_a.t / buffer_d.t;
				next[IDX(i,j,width)] = buffer_e.i;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep__texture__shared__(uint4 *next, int row, int col, int height, int width) {
	storage<T> buffer_a;
	storage<T> buffer_b;
	storage<T> buffer_c;
	storage<T> buffer_d;
	storage<T> buffer_e;

	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);

	buffer_d.i = tex1Dfetch(tex_w,IDX(row,col,width));

	__shared__ T __s__[SHARED_SIZE];

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < 1; j+= blockDim.y) {
			buffer_b.i = tex1Dfetch(tex_w,IDX(sharedIndexX+i,col,width));
			__s__[i] = buffer_b.t;
		}
	}
	for (int i = threadIdx.x; i < 1; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			buffer_c.i = tex1Dfetch(tex_w,IDX(row,sharedIndexY+j,width));
			__s__[SHARED_SIZE - j - 1] = buffer_c.t;
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j += blockDim.y) {
			if (sharedIndexX+i!=row && sharedIndexY+j!=col) {
				buffer_a.i = tex1Dfetch(tex_w,IDX(sharedIndexX+i,sharedIndexY+j,width));
				buffer_e.t = buffer_a.t - __s__[i]*__s__[SHARED_SIZE - j - 1]/ buffer_d.t;
				next[IDX(sharedIndexX+i,sharedIndexY+j,width)] = buffer_e.i;
			}
			else if (sharedIndexX+i==row && sharedIndexY+j!=col) {
				buffer_e.t = __s__[SHARED_SIZE - j - 1] / buffer_d.t;
				next[IDX(sharedIndexX+i,sharedIndexY+j,width)] = buffer_e.i;
			}
			else if (sharedIndexX+i!=row && sharedIndexY+j==col) {
				buffer_e.t = (T)0;
				next[IDX(sharedIndexX+i,sharedIndexY+j,width)] = buffer_e.i;
			}
			else {
				buffer_e.t = (T)1;
				next[IDX(sharedIndexX+i,sharedIndexY+j,width)] = buffer_e.i;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep2__texture__shared__(uint4 *next, int row, int col, int height, int width) {
	storage<T> buffer_a;
	storage<T> buffer_b;
	storage<T> buffer_c;
	storage<T> buffer_d;
	storage<T> buffer_e;

	int sharedIndexX = (int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedIndexY = (int)((width*(blockIdx.y+0))/gridDim.y);
	int sharedCountX = (int)((height*(blockIdx.x+1))/gridDim.x)-(int)((height*(blockIdx.x+0))/gridDim.x);
	int sharedCountY = (int)((width*(blockIdx.y+1))/gridDim.y)-(int)((width*(blockIdx.y+0))/gridDim.y);

	buffer_d.i = tex1Dfetch(tex_buffer,row);

	__shared__ T __s__[SHARED_SIZE];

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < 1; j+= blockDim.y) {
			buffer_b.i = tex1Dfetch(tex_buffer,sharedIndexX+i);
			__s__[i] = buffer_b.t;
		}
	}
	for (int i = threadIdx.x; i < 1; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j+= blockDim.y) {
			buffer_c.i = tex1Dfetch(tex_w,IDX(row,sharedIndexY+j,width));
			__s__[SHARED_SIZE - j - 1] = buffer_c.t;
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < sharedCountX; i += blockDim.x) {
		for (int j = threadIdx.y; j < sharedCountY; j += blockDim.y) {
			if (sharedIndexX+i!=row) {
				buffer_a.i = tex1Dfetch(tex_w,IDX(sharedIndexX+i,sharedIndexY+j,width));
				buffer_e.t = buffer_a.t - __s__[i]*__s__[SHARED_SIZE - j - 1]/ buffer_d.t;
				next[IDX(sharedIndexX+i,sharedIndexY+j,width)] = buffer_e.i;
			}
			else {
				buffer_e.t = __s__[SHARED_SIZE - j - 1] / buffer_d.t;
				next[IDX(sharedIndexX+i,sharedIndexY+j,width)] = buffer_e.i;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep__texture__local__(uint4 *next, int row, int col, int height, int width) {
	storage<T> buffer_a;
	storage<T> buffer_b;
	storage<T> buffer_c;
	storage<T> buffer_d;
	storage<T> buffer_e;

	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x))-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/(blockDim.x*gridDim.x));
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y))-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/(blockDim.y*gridDim.y));

	buffer_d.i = tex1Dfetch(tex_w,IDX(row,col,width));

	T __l__[LOCAL_SIZE];

	for (int i = 0; i < localCountX; i++) {
		for (int j = 0; j < 1; j++) {
			buffer_b.i = tex1Dfetch(tex_w,IDX(localIndexX+i,col,width));
			__l__[i] = buffer_b.t;
		}
	}
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < localCountY; j++) {
			buffer_c.i = tex1Dfetch(tex_w,IDX(row,localIndexY+j,width));
			__l__[LOCAL_SIZE - j - 1] = buffer_c.t;
		}
	}

	for (int i = 0; i < localCountX; i++ ) {
		for (int j = 0; j < localCountY; j++ ) {
			if (localIndexX+i!=row && localIndexY+j!=col) {
				buffer_a.i = tex1Dfetch(tex_w,IDX(localIndexX+i,localIndexY+j,width));
				buffer_e.t = buffer_a.t - __l__[i]*__l__[LOCAL_SIZE - j - 1]/ buffer_d.t;
				next[IDX(localIndexX+i,localIndexY+j,width)] = buffer_e.i;
			}
			else if (localIndexX+i==row && localIndexY+j!=col) {
				buffer_e.t = __l__[LOCAL_SIZE - j - 1] / buffer_d.t;
				next[IDX(localIndexX+i,localIndexY+j,width)] = buffer_e.i;
			}
			else if (localIndexX+i!=row && localIndexY+j==col) {
				buffer_e.t = (T)0;
				next[IDX(localIndexX+i,localIndexY+j,width)] = buffer_e.i;
			}
			else {
				buffer_e.t = (T)1;
				next[IDX(localIndexX+i,localIndexY+j,width)] = buffer_e.i;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_gaussjordanstep2__texture__local__(uint4 *next, int row, int col, int height, int width) {
	storage<T> buffer_a;
	storage<T> buffer_b;
	storage<T> buffer_c;
	storage<T> buffer_d;
	storage<T> buffer_e;

	int localIndexX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x));
	int localIndexY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y));
	int localCountX = (int)((height*(blockDim.x*blockIdx.x+threadIdx.x+0))/(blockDim.x*gridDim.x))-(int)((height*(blockDim.x*blockIdx.x+threadIdx.x+1))/(blockDim.x*gridDim.x));
	int localCountY = (int)((width*(blockDim.y*blockIdx.y+threadIdx.y+0))/(blockDim.y*gridDim.y))-(int)((width*(blockDim.y*blockIdx.y+threadIdx.y+1))/(blockDim.y*gridDim.y));

	buffer_d.i = tex1Dfetch(tex_buffer,row);

	T __l__[LOCAL_SIZE];

	for (int i = 0; i < localCountX; i++) {
		for (int j = 0; j < 1; j++) {
			buffer_b.i = tex1Dfetch(tex_buffer,localIndexX+i);
			__l__[i] = buffer_b.t;
		}
	}
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < localCountY; j++) {
			buffer_c.i = tex1Dfetch(tex_w,IDX(row,localIndexY+j,width));
			__l__[LOCAL_SIZE - j - 1] = buffer_c.t;
		}
	}

	for (int i = 0; i < localCountX; i++ ) {
		for (int j = 0; j < localCountY; j++ ) {
			if (localIndexX+i!=row) {
				buffer_a.i = tex1Dfetch(tex_w,IDX(localIndexX+i,localIndexY+j,width));
				buffer_e.t = buffer_a.t - __l__[i]*__l__[LOCAL_SIZE - j - 1]/ buffer_d.t;
				next[IDX(localIndexX+i,localIndexY+j,width)] = buffer_e.i;
			}
			else {
				buffer_e.t = __l__[LOCAL_SIZE - j - 1] / buffer_d.t;
				next[IDX(localIndexX+i,localIndexY+j,width)] = buffer_e.i;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_find_first_notzero__global__(T *a, int *b, int height, int width, T tolerance, size_t pitch) {
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		b[i] = -1;
		for(int index = 0;index<width;index++) {
			T x = ELEMENT(T,a,i,index,pitch);
			if (abs(x)>-tolerance) {
				b[i] = index;
				break;
			}
		}
	}
}

template<class T> __global__ void __global__matrix_find_first_notzero__texture__(int *b, int height, int width, T tolerance) {
	storage<T> buffer;
	for (int i = blockDim.x*blockIdx.x + threadIdx.x; i < height; i += blockDim.x*gridDim.x) {
		b[i] = -1;
		for(int index = 0;index<width;index++) {
			buffer.i = tex1Dfetch(tex_w,IDX(i,index,width));
			if (abs(buffer.t)>-tolerance) {
				b[i] = index;
				break;
			}
		}
	}
}

template<class T> __host__ void __cdecl __host__matrix_gaussjordanstep(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, int row, int col, MEMORY src, MEMORY dest, MEMORY cache) {
	T *d_w[2];
	size_t pitch;
	cudaError_t err;
	storage<T> buffer;

	int height = a->height;
	int width =  a->width;
	b->height = height;
	b->width = width;

	switch(src) {
	case TEXTURE:
		err = cudaMallocHost((void**)&cpu_wv, (size_t) width * height * sizeof(uint4));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				buffer.t = a->values[IDX(i,j,width)];
				cpu_wv[IDX(i,j,width)] = buffer.i;
			}
		}
		for(int i=0; i<2; i++) {
			err = cudaMalloc((void**)&gpu_w[i], (size_t) width * height * sizeof(uint4));
		}
		//  настройка параемтров текстуры  texture
		tex_w.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_w.addressMode[1] = cudaAddressModeWrap;
		tex_w.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_w.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaMemcpy((void*)gpu_w[0], (void*)cpu_wv, (size_t) width * height * sizeof(uint4), cudaMemcpyHostToDevice);
		break;
	default:
		for(int i=0; i<2; i++) {
			err = cudaMallocPitch((void**)&d_w[i], &pitch, (size_t) width * sizeof(T), (size_t) height);
		}
		err = cudaMemcpy2D((void*)d_w[0], pitch, (void*)a->values, (size_t) a->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		break;
	}

	if(src == GLOBAL && cache == NONE) {
		__global__matrix_gaussjordanstep__global__none__<T><<<blocks,threads>>>(d_w[0],d_w[1], row, col, height, width, pitch);
	}
	else if(src == GLOBAL && cache == SHARED) {
		__global__matrix_gaussjordanstep__global__shared__<T><<<blocks,threads>>>(d_w[0],d_w[1], row, col, height, width, pitch);
	}
	else if(src == GLOBAL && cache == LOCAL) {
		__global__matrix_gaussjordanstep__global__local__<T><<<blocks,threads>>>(d_w[0],d_w[1], row, col, height, width, pitch);
	}
	else if(src == TEXTURE && cache == NONE) {
		err = cudaBindTexture(0, tex_w, gpu_w[0], (size_t) width * height * sizeof(uint4));
		__global__matrix_gaussjordanstep__texture__none__<T><<<blocks,threads>>>(gpu_w[1], row, col, height, width);
		err = cudaUnbindTexture(tex_w);
	}
	else if(src == TEXTURE && cache == SHARED) {
		err = cudaBindTexture(0, tex_w, gpu_w[0], (size_t) width * height * sizeof(uint4));
		__global__matrix_gaussjordanstep__texture__shared__<T><<<blocks,threads>>>(gpu_w[1], row, col, height, width);
		err = cudaUnbindTexture(tex_w);
	}
	else if(src == TEXTURE && cache == LOCAL) {
		err = cudaBindTexture(0, tex_w, gpu_w[0], (size_t) width * height * sizeof(uint4));
		__global__matrix_gaussjordanstep__texture__local__<T><<<blocks,threads>>>(gpu_w[1], row, col, height, width);
		err = cudaUnbindTexture(tex_w);
	}

	switch(src) {
	case TEXTURE:
		err = cudaMemcpy((void*)cpu_wv, (void*)gpu_w[1], (size_t) width * height * sizeof(uint4), cudaMemcpyDeviceToHost);
		for(int i=0; i<height; i++) {
			for(int j=0; j<width; j++) {
				b->values[IDX(i,j,width)] = *(T*)&cpu_wv[IDX(i,j,width)];
			}
		}
		for(int i=0; i<2; i++) {
			err = cudaFree(gpu_w[i]);
		}
		err = cudaFreeHost((void*)cpu_wv);
		break;
	default:
		err = cudaMemcpy2D((void*)b->values, (size_t) b->width * sizeof(T), (void*)d_w[1], pitch, (size_t) width * sizeof(T), (size_t) height, cudaMemcpyDeviceToHost);
		for(int i=0; i<2; i++) {
			err = cudaFree(d_w[i]);
		}
		break;
	}

	err = err;
}

template<class T> __host__ void __cdecl __host__matrix_gaussjordan(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, T tolerance, MEMORY src, MEMORY dest, MEMORY cache) {
	T *d_w[2];
	int *h_index;
	int *d_index;
	size_t pitch;
	cudaError_t err;
	storage<T> buffer;

	int height = a->height;
	int width =  a->width;
	b->height = height;
	b->width = width;

	err = cudaMallocHost((void**)&h_index, height*sizeof(int));
	err = cudaMalloc((void**)&d_index, height*sizeof(int));

	switch(src) {
	case TEXTURE:
		err = cudaMallocHost((void**)&cpu_wv, (size_t) width * height * sizeof(uint4));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				buffer.t = a->values[IDX(i,j,width)];
				cpu_wv[IDX(i,j,width)] = buffer.i;
			}
		}
		for(int i=0; i<2; i++) {
			err = cudaMalloc((void**)&gpu_w[i], (size_t) width * height * sizeof(uint4));
		}
		//  настройка параемтров текстуры  texture
		tex_w.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_w.addressMode[1] = cudaAddressModeWrap;
		tex_w.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_w.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaMemcpy((void*)gpu_w[0], (void*)cpu_wv, (size_t) width * height * sizeof(uint4), cudaMemcpyHostToDevice);
		break;
	default:
		for(int i=0; i<2; i++) {
			err = cudaMallocPitch((void**)&d_w[i], &pitch, (size_t) width * sizeof(T), (size_t) height);
		}
		err = cudaMemcpy2D((void*)d_w[0], pitch, (void*)a->values, (size_t) a->width * sizeof(T), (size_t) width * sizeof(T), (size_t) height, cudaMemcpyHostToDevice);
		break;
	}
	
	for(int k=0;;k++) {
		if(src == GLOBAL) {
			__global__matrix_find_first_notzero__global__<T><<<1,height>>>(d_w[k&1],d_index,height,width,tolerance,pitch);
		}
		else if(src == TEXTURE) {
			err = cudaBindTexture(0, tex_w, gpu_w[k&1], (size_t) width * height * sizeof(uint4));
			__global__matrix_find_first_notzero__texture__<T><<<1,height>>>(d_index,height,width,tolerance);
		}
		cudaMemcpy((void*)h_index,(void*)d_index,(size_t) height*sizeof(int),cudaMemcpyDeviceToHost);
		while(h_index[k]<0 && k<min(height,width)) k++;
		if(k == min(height,width)) {
			switch(src) {
			case TEXTURE:
				err = cudaUnbindTexture(tex_w);
				err = cudaMemcpy((void*)cpu_wv, (void*)gpu_w[k&1], (size_t) width * height * sizeof(uint4), cudaMemcpyDeviceToHost);
				for(int i=0; i<height; i++) {
					for(int j=0; j<width; j++) {
						b->values[IDX(i,j,width)] = *(T*)&cpu_wv[IDX(i,j,width)];
					}
				}
				break;
			case GLOBAL:
			default:
				err = cudaMemcpy2D((void*)b->values, (size_t) b->width * sizeof(T), (void*)d_w[k&1], pitch, (size_t) width * sizeof(T), (size_t) height, cudaMemcpyDeviceToHost);
				break;
			}
			break;
		}
		if(src == GLOBAL && cache == NONE) {
			__global__matrix_gaussjordanstep__global__none__<T><<<blocks,threads>>>(d_w[k&1],d_w[1-(k&1)], k, h_index[k], height, width, pitch);
		}
		else if(src == GLOBAL && cache == SHARED) {
			__global__matrix_gaussjordanstep__global__shared__<T><<<blocks,threads>>>(d_w[k&1],d_w[1-(k&1)], k, h_index[k], height, width, pitch);
		}
		else if(src == GLOBAL && cache == LOCAL) {
			__global__matrix_gaussjordanstep__global__local__<T><<<blocks,threads>>>(d_w[k&1],d_w[1-(k&1)], k, h_index[k], height, width, pitch);
		}
		else if(src == TEXTURE && cache == NONE) {
			__global__matrix_gaussjordanstep__texture__none__<T><<<blocks,threads>>>(gpu_w[1-(k&1)], k, h_index[k], height, width);
			err = cudaUnbindTexture(tex_w);
		}
		else if(src == TEXTURE && cache == SHARED) {
			__global__matrix_gaussjordanstep__texture__shared__<T><<<blocks,threads>>>(gpu_w[1-(k&1)], k, h_index[k], height, width);
			err = cudaUnbindTexture(tex_w);
		}
		else if(src == TEXTURE && cache == LOCAL) {
			__global__matrix_gaussjordanstep__texture__local__<T><<<blocks,threads>>>(gpu_w[1-(k&1)], k, h_index[k], height, width);
			err = cudaUnbindTexture(tex_w);
		}
	}

	switch(src) {
	case TEXTURE:
		for(int i=0; i<2; i++) {
			err = cudaFree(gpu_w[i]);
		}
		err = cudaFreeHost((void*)cpu_wv);
		break;
	default:
		for(int i=0; i<2; i++) {
			err = cudaFree(d_w[i]);
		}
		break;
	}

	err = cudaFree(d_index);
	err = cudaFreeHost((void*)h_index);

	err = err;
}

// Вычисление обратной матрицы
template<class T> __host__ void __cdecl __host__matrix_inv(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, T tolerance,  MEMORY src, MEMORY dest, MEMORY cache) {
	T *d_w[2];
	T *d_v[2];
	int *h_index;
	int *d_index;
	size_t pitch;
	T * d_buffer;
	cudaError_t err;
	storage<T> buffer;

	int height = a->height;
	int width =  a->width;
	b->height = height;
	b->width = width;
	
	assert(a->height == a->width);

	if (height==1 && width==1) {
		b->values[0] = (T)1/a->values[0];
		return;
	}

	err = cudaMallocHost((void**)&h_index, (size_t) width * sizeof(int));
	err = cudaMalloc((void**)&d_index, (size_t) width * sizeof(int));

	switch(src) {
	case TEXTURE:
		err = cudaMallocHost((void**)&cpu_wv, (size_t) width * width * sizeof(uint4));
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				buffer.t = a->values[IDX(i,j,width)];
				cpu_wv[IDX(i,j,width)] = buffer.i;
			}
		}
		for(int i=0; i<2; i++) {
			err = cudaMalloc((void**)&gpu_w[i], (size_t) width * width * sizeof(uint4));
			err = cudaMalloc((void**)&gpu_v[i], (size_t) width * width * sizeof(uint4));
		}
		//  настройка параемтров текстуры  texture
		tex_w.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_w.addressMode[1] = cudaAddressModeWrap;
		tex_w.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_w.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaMemcpy((void*)gpu_w[0], (void*)cpu_wv, (size_t) width * width * sizeof(uint4), cudaMemcpyHostToDevice);
		//  настройка параемтров текстуры  texture
		tex_v.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_v.addressMode[1] = cudaAddressModeWrap;
		tex_v.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_v.normalized     = false;                // не использовать нормализованную адресацию
		for(int i=0; i<a->height; i++) {
			for(int j=0; j<a->width; j++) {
				buffer.t = (i==j)?(T)1:(T)0;
				cpu_wv[IDX(i,j,width)] = buffer.i;
			}
		}
		err = cudaMemcpy((void*)gpu_v[0], (void*)cpu_wv, (size_t) width * width * sizeof(uint4), cudaMemcpyHostToDevice);
		err = cudaMalloc((void**)&gpu_buffer, (size_t) width * sizeof(uint4));
		//  настройка параемтров текстуры  texture
		tex_buffer.addressMode[0] = cudaAddressModeWrap;  // режим Wrap
		tex_buffer.addressMode[1] = cudaAddressModeWrap;
		tex_buffer.filterMode     = cudaFilterModePoint;  // ближайшее значение
		tex_buffer.normalized     = false;                // не использовать нормализованную адресацию
		err = cudaBindTexture(0, tex_buffer, gpu_buffer, (size_t) width * width * sizeof(uint4));
		break;
	default:
		for(int i=0; i<2; i++) {
			err = cudaMallocPitch((void**)&d_w[i], &pitch, (size_t) width * sizeof(T), (size_t) width);
			err = cudaMallocPitch((void**)&d_v[i], &pitch, (size_t) width * sizeof(T), (size_t) width);
		}
		err = cudaMemcpy2D((void*)d_w[0], pitch, (void*)a->values, (size_t) width * sizeof(T), (size_t) width * sizeof(T), width, cudaMemcpyHostToDevice);
		for(int i=0; i<height; i++) {
			for(int j=0; j<width; j++) {
				b->values[IDX(i,j,width)] = (i==j)?(T)1:(T)0;
			}
		}
		err = cudaMemcpy2D((void*)d_v[0], pitch, (void*)b->values, (size_t) width * sizeof(T), (size_t) width * sizeof(T), width, cudaMemcpyHostToDevice);
		err = cudaMalloc((void**)&d_buffer, (size_t) width * sizeof(T));
		break;
	}

	for(int k=0;;k++) {
		if(src == GLOBAL) {
			__global__matrix_find_first_notzero__global__<T><<<1,width>>>(d_w[k&1],d_index,width,width,tolerance,pitch);
		}
		else if(src == TEXTURE) {
			err = cudaBindTexture(0, tex_w, gpu_w[k&1], (size_t) width * width * sizeof(uint4));
			__global__matrix_find_first_notzero__texture__<T><<<1,width>>>(d_index,width,width,tolerance);
		}
		err = cudaMemcpy((void*)h_index,(void*)d_index,(size_t) width * sizeof(int),cudaMemcpyDeviceToHost);
		while(h_index[k]<0 && k<min(height,width)) k++;
		if(k == min(height,width)) {
			switch(src) {
			case GLOBAL:
				err = cudaMemcpy2D((void*)b->values,width*sizeof(T),(void*)d_v[k&1],pitch,width*sizeof(T),width,cudaMemcpyDeviceToHost);
				break;
			case TEXTURE:
				err = cudaUnbindTexture(tex_w);
				err = cudaMemcpy((void*)cpu_wv,(void*)gpu_v[k&1],width*width*sizeof(uint4),cudaMemcpyDeviceToHost);
				for(int i=0; i<height; i++) {
					for(int j=0; j<width; j++) {
						buffer.i=cpu_wv[IDX(i,j,width)];
						b->values[IDX(i,j,width)] = buffer.t;
					}
				}
				break;
			}
			break;
		}
		if(src == GLOBAL && cache == NONE) {
			err = cudaMemcpy2D((void*)d_buffer, 1*sizeof(T), (void*)&ELEMENT(T,d_w[k&1],0,h_index[k],pitch), pitch, (size_t) 1 * sizeof(T), width, cudaMemcpyDeviceToDevice);
			__global__matrix_gaussjordanstep__global__none__<T><<<blocks,threads>>>(d_w[k&1],d_w[1-(k&1)], k, h_index[k], width, width, pitch);
			__global__matrix_gaussjordanstep2__global__none__<T><<<blocks,threads>>>(d_buffer,d_v[k&1],d_v[1-(k&1)], k, h_index[k], width, width, pitch);
		}
		else if(src == GLOBAL && cache == SHARED) {
			err = cudaMemcpy2D((void*)d_buffer, 1*sizeof(T), (void*)&ELEMENT(T,d_w[k&1],0,h_index[k],pitch), pitch, (size_t) 1 * sizeof(T), width, cudaMemcpyDeviceToDevice);
			__global__matrix_gaussjordanstep__global__shared__<T><<<blocks,threads>>>(d_w[k&1],d_w[1-(k&1)], k, h_index[k], width, width, pitch);
			__global__matrix_gaussjordanstep2__global__shared__<T><<<blocks,threads>>>(d_buffer,d_v[k&1],d_v[1-(k&1)], k, h_index[k], width, width, pitch);
		}
		else if(src == GLOBAL && cache == LOCAL) {
			err = cudaMemcpy2D((void*)d_buffer, 1*sizeof(T), (void*)&ELEMENT(T,d_w[k&1],0,h_index[k],pitch), pitch, (size_t) 1 * sizeof(T), width, cudaMemcpyDeviceToDevice);
			__global__matrix_gaussjordanstep__global__local__<T><<<blocks,threads>>>(d_w[k&1],d_w[1-(k&1)], k, h_index[k], width, width, pitch);
			__global__matrix_gaussjordanstep2__global__local__<T><<<blocks,threads>>>(d_buffer,d_v[k&1],d_v[1-(k&1)], k, h_index[k], width, width, pitch);
		}
		else if(src == TEXTURE && cache == NONE) {
			err = cudaMemcpy2D((void*)gpu_buffer, 1*sizeof(uint4), (void*)&gpu_w[k&1][IDX(0,h_index[k],width)], width*sizeof(uint4), (size_t) 1 * sizeof(uint4), width, cudaMemcpyDeviceToDevice);
			err = cudaBindTexture(0, tex_v, gpu_v[k&1], (size_t) width * width * sizeof(uint4));
			__global__matrix_gaussjordanstep__texture__none__<T><<<blocks,threads>>>(gpu_w[1-(k&1)], k, h_index[k], width, width);
			__global__matrix_gaussjordanstep2__texture__none__<T><<<blocks,threads>>>(gpu_v[1-(k&1)], k, h_index[k], width, width);
			err = cudaUnbindTexture(tex_w);
			err = cudaUnbindTexture(tex_v);
		}
		else if(src == TEXTURE && cache == SHARED) {
			err = cudaMemcpy2D((void*)gpu_buffer, 1*sizeof(uint4), (void*)&gpu_w[k&1][IDX(0,h_index[k],width)], width*sizeof(uint4), (size_t) 1 * sizeof(uint4), width, cudaMemcpyDeviceToDevice);
			err = cudaBindTexture(0, tex_v, gpu_v[k&1], (size_t) width * width * sizeof(uint4));
			__global__matrix_gaussjordanstep__texture__shared__<T><<<blocks,threads>>>(gpu_w[1-(k&1)], k, h_index[k], width, width);
			__global__matrix_gaussjordanstep2__texture__shared__<T><<<blocks,threads>>>(gpu_v[1-(k&1)], k, h_index[k], width, width);
			err = cudaUnbindTexture(tex_w);
			err = cudaUnbindTexture(tex_v);
		}
		else if(src == TEXTURE && cache == LOCAL) {
			err = cudaMemcpy2D((void*)gpu_buffer, 1*sizeof(uint4), (void*)&gpu_w[k&1][IDX(0,h_index[k],width)], width*sizeof(uint4), (size_t) 1 * sizeof(uint4), width, cudaMemcpyDeviceToDevice);
			err = cudaBindTexture(0, tex_v, gpu_v[k&1], (size_t) width * width * sizeof(uint4));
			__global__matrix_gaussjordanstep__texture__local__<T><<<blocks,threads>>>(gpu_w[1-(k&1)], k, h_index[k], width, width);
			__global__matrix_gaussjordanstep2__texture__local__<T><<<blocks,threads>>>(gpu_v[1-(k&1)], k, h_index[k], width, width);
			err = cudaUnbindTexture(tex_w);
			err = cudaUnbindTexture(tex_v);
		}
	}
	
	switch(src) {
	case TEXTURE:
		for(int i=0; i<2; i++) {
			err = cudaFree(gpu_w[i]);
			err = cudaFree(gpu_v[i]);
		}
		err = cudaFree(cpu_wv);
		err = cudaFree(gpu_buffer);
		break;
	default:
		for(int i=0; i<2; i++) {
			err = cudaFree(d_w[i]);
			err = cudaFree(d_v[i]);
		}
		err = cudaFree(d_buffer);
		break;
	}

	err = cudaFree(d_index);
	err = cudaFreeHost(h_index);

	err = err;
}

template __host__ void __cdecl __host__matrix_gaussjordanstep<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, int row, int col, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __cdecl __host__matrix_gaussjordan<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, double tolerance, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __cdecl __host__matrix_inv<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, double tolerance, MEMORY src, MEMORY dest, MEMORY cache);
