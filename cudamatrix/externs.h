#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>

extern template __host__ void __cdecl __host__matrix_plus<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_minus<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b,MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_rot<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_inv<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, DATATYPE tolerance, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_add<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, MATRIX<DATATYPE> *c, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_sub<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, MATRIX<DATATYPE> *c, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_mul<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, MATRIX<DATATYPE> *c, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_gaussjordanstep<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, int row, int col, MEMORY src, MEMORY dest, MEMORY cache);
extern template __host__ void __cdecl __host__matrix_gaussjordan<DATATYPE>(dim3 blocks, dim3 threads, MATRIX<DATATYPE> *a, MATRIX<DATATYPE> *b, DATATYPE tolerance, MEMORY src, MEMORY dest, MEMORY cache);
extern template void __cdecl matrix_read<DATATYPE>(char *fileName, MATRIX<DATATYPE> ** matrix);
extern template void __cdecl matrix_write<DATATYPE>(char *fileName, MATRIX<DATATYPE> * matrix);

extern char * __cdecl string_stack[1024];
extern int __cdecl string_stack_size;
extern DATATYPE __cdecl double_stack[1024];
extern int __cdecl double_stack_size;
