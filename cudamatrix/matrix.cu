#define __NO_EXTERN__
#include "matrix.h"

// Constant memory (64KB max) can only be hard-coded before compilation.
__constant__ char __cdecl __c__[CONST_SIZE];

texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_a;
texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_b;
texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_w;
texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_v;
texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_buffer;

uint4 * __cdecl gpu_a;
uint4 * __cdecl gpu_b;
uint4 * __cdecl cpu_a;
uint4 * __cdecl cpu_b;
uint4 * __cdecl gpu_w[2];
uint4 * __cdecl gpu_v[2];
uint4 * __cdecl cpu_wv;
uint4 * __cdecl gpu_buffer;

char * __cdecl string_stack[1024];
int __cdecl string_stack_size = 0;

double __cdecl double_stack[1024];
int __cdecl double_stack_size = 0;

