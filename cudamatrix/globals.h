#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>

// The first point is that in the classic CUDA compilation model, 
// the scope requirement for all device symbols is translation unit scope. And all means at least
// __global__ functions
// __device__ functions
// __texture__ objects
// __constant__ memory
// __shared__ memory declared extern
// __device__ symbols
// All of these have to be defined at the same translation unit where they are used in order
// for the runtime to work correctly. The underlying reason is that all of the objects
// on the list above must get emitted into the same CUDA module (this is a driver API concept)
// for everything to work.

// CUDA 5 introduces a separate compilation mode, which is effectively a static linkage model. 
// This means symbols don't need defined in the same translation unit, they can be forward declared 
// and the linker will handle symbol matching during a device code linkage pass.

__constant__ uint4 __cdecl __c__[CONST_SIZE];

extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_a;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_b;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_w;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_v;
extern texture<uint4, cudaTextureType1D, cudaReadModeElementType> __cdecl tex_buffer;
extern uint4 * __cdecl gpu_a;
extern uint4 * __cdecl gpu_b;
extern uint4 * __cdecl cpu_a;
extern uint4 * __cdecl cpu_b;
extern uint4 * __cdecl gpu_w[];
extern uint4 * __cdecl gpu_v[];
extern uint4 * __cdecl cpu_wv;
extern uint4 * __cdecl gpu_buffer;
