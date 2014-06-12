#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>

extern __constant__ char __c__[CONST_SIZE];

extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_a;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_b;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_w;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_v;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_buffer;
extern uint4 * __cdecl gpu_a;
extern uint4 * __cdecl gpu_b;
extern uint4 * __cdecl cpu_a;
extern uint4 * __cdecl cpu_b;
extern uint4 * __cdecl gpu_w[2];
extern uint4 * __cdecl gpu_v[2];
extern uint4 * __cdecl cpu_wv;
extern uint4 * __cdecl gpu_buffer;
