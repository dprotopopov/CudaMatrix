#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>

extern template __host__ void __cdecl __host__matrix_plus<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_minus<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b,MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_rot<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_inv<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, double tolerance, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_add<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MATRIX<double> *c, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_sub<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MATRIX<double> *c, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_mul<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, MATRIX<double> *c, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_gaussjordanstep<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, int row, int col, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_gaussjordan<double>(dim3 blocks, dim3 threads, MATRIX<double> *a, MATRIX<double> *b, double tolerance, MEMORY src, MEMORY dest, MEMORY cache);
extern template void __cdecl matrix_read<double>(char *fileName, MATRIX<double> ** matrix);
extern template void __cdecl matrix_write<double>(char *fileName, MATRIX<double> * matrix);

extern char * __cdecl string_stack[1024];
extern int __cdecl string_stack_size;
extern double __cdecl double_stack[1024];
extern int __cdecl double_stack_size;
