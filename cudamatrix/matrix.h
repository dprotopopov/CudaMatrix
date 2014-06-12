#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>

#define CONST_SIZE	(1<<10)
#define SHARED_SIZE	(1<<10)
#define LOCAL_SIZE	(1<<10)

#define IDX(i,j,n) ((i)*(n)+j)

// The pitch returned in *pitch by cudaMallocPitch() is the width in bytes of the allocation. 
// The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses
// within the 2D array. Given the row and column of an array element of type T, the address is computed as: 
//‎    T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
// For allocations of 2D arrays, it is recommended that programmers consider performing pitch allocations 
// using cudaMallocPitch(). Due to pitch alignment restrictions in the hardware, this is especially true
// if the application will be performing 2D memory copies between different regions of device memory
// (whether linear memory or CUDA arrays). 

#define ELEMENT(T,BaseAddress,Row,Column,Pitch) *(T*)((char*)(BaseAddress) + (Row) * (Pitch) + (Column*sizeof(T)))

#define assert( bool ) 

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef abs
#define abs( a ) max( (a) , (-a) )
#endif

template<class T> struct MATRIX {
	int height;
	int width;
	T values[4];
};

enum MEMORY {
	NONE,
	LOCAL,
	SHARED,
	GLOBAL,
	CONSTANT,
	TEXTURE
};

template<class T> __host__ void __host__matrix_plus(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_minus(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b,MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_rot(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_inv(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, T tolerance, MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_add(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MATRIX<T> *c, MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_sub(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MATRIX<T> *c, MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_mul(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MATRIX<T> *c, MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_div(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, MATRIX<T> *c, T tolerance, MEMORY src, MEMORY dest, MEMORY cache);

template<class T> __host__ void __host__matrix_gaussjordanstep(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, int row, int col, MEMORY src, MEMORY dest, MEMORY cache);
template<class T> __host__ void __host__matrix_gaussjordan(dim3 blocks, dim3 threads, MATRIX<T> *a, MATRIX<T> *b, T tolerance, MEMORY src, MEMORY dest, MEMORY cache);

template<class T> void matrix_read(char *fileName, MATRIX<T> ** matrix);
template<class T> void matrix_write(char *fileName, MATRIX<T> * matrix);

extern char * __cdecl string_stack[1024];
extern int __cdecl string_stack_size;

extern double __cdecl double_stack[1024];
extern int __cdecl double_stack_size;

template __host__ void __host__matrix_plus<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_minus<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b,MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_rot<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_inv<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, double tolerance, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_add<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MATRIX<double> *c, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_sub<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MATRIX<double> *c, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_mul<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MATRIX<double> *c, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_div<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MATRIX<double> *c, double tolerance, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_gaussjordanstep<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, int row, int col, MEMORY src, MEMORY dest, MEMORY cache);
template __host__ void __host__matrix_gaussjordan<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, double tolerance, MEMORY src, MEMORY dest, MEMORY cache);
template void matrix_read<double>(char *fileName, MATRIX<double> ** matrix);
template void matrix_write<double>(char *fileName, MATRIX<double> * matrix);


