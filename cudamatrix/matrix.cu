#include "matrix.h"

texture<TEXTURESTORAGETYPE, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_a;
texture<TEXTURESTORAGETYPE, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_b;
texture<TEXTURESTORAGETYPE, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_w;
texture<TEXTURESTORAGETYPE, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_v;
texture<TEXTURESTORAGETYPE, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_buffer;

TEXTURESTORAGETYPE * __cdecl gpu_a;
TEXTURESTORAGETYPE * __cdecl gpu_b;
TEXTURESTORAGETYPE * __cdecl cpu_a;
TEXTURESTORAGETYPE * __cdecl cpu_b;
TEXTURESTORAGETYPE * __cdecl gpu_w[2];
TEXTURESTORAGETYPE * __cdecl gpu_v[2];
TEXTURESTORAGETYPE * __cdecl cpu_wv;
TEXTURESTORAGETYPE * __cdecl gpu_buffer;

char * __cdecl string_stack[1024];
int __cdecl string_stack_size = 0;

DATATYPE __cdecl double_stack[1024];
int __cdecl double_stack_size = 0;

