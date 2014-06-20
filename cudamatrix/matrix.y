%{
#include "matrix.h"
#include "externs.h"

void yyerror(char *);
int yylex(void);
MATRIX<DATATYPE> * sym[26];
MEMORY source_memory = TEXTURE;
MEMORY dest_memory = GLOBAL;
MEMORY cache_memory = SHARED;
dim3 blocks = dim3(1,1,1);
dim3 threads = dim3(1,1,1);
DATATYPE tolerance = 0;

%}
%token SHOW_KEYWORD INFO_KEYWORD HELP_KEYWORD
%token READ_KEYWORD WRITE_KEYWORD ECHO_KEYWORD
%token LET_KEYWORD SET_KEYWORD USE_KEYWORD
%token ALLOC_KEYWORD FREE_KEYWORD 
%token SRC_KEYWORD DEST_KEYWORD CACHE_KEYWORD
%token INTEGER DOUBLE VARIABLE STRING
%token TEXTURE_KEYWORD CONSTANT_KEYWORD GLOBAL_KEYWORD SHARED_KEYWORD LOCAL_KEYWORD NONE_KEYWORD
%token BLOCKS_KEYWORD THREADS_KEYWORD TOLERANCE_KEYWORD
%token GAUSSJORDAN_KEYWORD ROT_KEYWORD INV_KEYWORD
%left '+' '-' '*'
%%
program:
program statement '\n'
	| /* NULL */
	;
statement:
	ECHO_KEYWORD STRING { printf("%s\n",string_stack[$2]); }
	| LET_KEYWORD VARIABLE '=' VARIABLE { 
			if ($3==$4) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[$4]->height;
			int width = sym[$4]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			memcpy(sym[$4]->values,sym[$2]->values,height*width*sizeof(DATATYPE)); 
		}
	| LET_KEYWORD VARIABLE '=' '+' VARIABLE { 
			if ($2==$5) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[$5]->height;
			int width = sym[$5]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_plus<DATATYPE>(blocks,threads,sym[$5],sym[$2],source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' '-' VARIABLE { 
			if ($2==$5) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[$5]->height;
			int width = sym[$5]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_minus<DATATYPE>(blocks,threads,sym[$5],sym[$2],source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' ROT_KEYWORD VARIABLE { 
			if ($2==$5) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[$5]->height;
			int width = sym[$5]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_rot<DATATYPE>(blocks,threads,sym[$5],sym[$2],source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' INV_KEYWORD VARIABLE { 
			if ($2==$5) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[$5]->height;
			int width = sym[$5]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_inv<DATATYPE>(blocks,threads,sym[$5],sym[$2],tolerance,source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' GAUSSJORDAN_KEYWORD INTEGER INTEGER VARIABLE { 
			if ($2==$7) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[$7]->height;
			int width = sym[$7]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_gaussjordanstep<DATATYPE>(blocks,threads,sym[$7],sym[$1],$5,$6,source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' GAUSSJORDAN_KEYWORD VARIABLE { 
			if ($2==$5) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[$5]->height;
			int width = sym[$5]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_gaussjordan<DATATYPE>(blocks,threads,sym[$5],sym[$2],tolerance,source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' VARIABLE '+' VARIABLE { 
			if ($2==$4) { yyerror("L-value must be different from R-value"); exit(-1); }
			if ($2==$6) { yyerror("L-value must be different from R-value"); exit(-1); }
			if (sym[$4]->height!=sym[$6]->height) { yyerror("First argument height must be equal second argument height"); exit(-1); }
			if (sym[$4]->width!=sym[$6]->width) { yyerror("First argument width must be equal second argument width"); exit(-1); }
			int height = sym[$4]->height;
			int width = sym[$6]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_add<DATATYPE>(blocks,threads,sym[$4],sym[$6],sym[$2],source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' VARIABLE '-' VARIABLE { 
			if ($2==$4) { yyerror("L-value must be different from R-value"); exit(-1); }
			if ($2==$6) { yyerror("L-value must be different from R-value"); exit(-1); }
			if (sym[$4]->height!=sym[$6]->height) { yyerror("First argument height must be equal second argument height"); exit(-1); }
			if (sym[$4]->width!=sym[$6]->width) { yyerror("First argument width must be equal second argument width"); exit(-1); }
			int height = sym[$4]->height;
			int width = sym[$6]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_sub<DATATYPE>(blocks,threads,sym[$4],sym[$6],sym[$2],source_memory,dest_memory,cache_memory); 
		}
	| LET_KEYWORD VARIABLE '=' VARIABLE '*' VARIABLE { 
			if ($2==$4) { yyerror("L-value must be different from R-value"); exit(-1); }
			if ($2==$6) { yyerror("L-value must be different from R-value"); exit(-1); }
			if (sym[$4]->width!=sym[$6]->height) { yyerror("First argument width must be equal second argument height"); exit(-1); }
			int height = sym[$4]->height;
			int width_height = sym[$4]->width;
			int width = sym[$6]->width;
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+(width * height * sizeof(DATATYPE)));
			sym[$2]->height = height;
			sym[$2]->width = width;
			__host__matrix_mul<DATATYPE>(blocks,threads,sym[$4],sym[$6],sym[$2],source_memory,dest_memory,cache_memory); 
		}
	| READ_KEYWORD VARIABLE STRING { matrix_read<DATATYPE>(string_stack[$3], &sym[$2]); }
	| WRITE_KEYWORD VARIABLE STRING { matrix_write<DATATYPE>(string_stack[$3], sym[$2]); }
	| USE_KEYWORD SRC_KEYWORD TEXTURE_KEYWORD { source_memory = TEXTURE; }
	| USE_KEYWORD SRC_KEYWORD CONSTANT_KEYWORD { source_memory = CONSTANT; }
	| USE_KEYWORD SRC_KEYWORD GLOBAL_KEYWORD { source_memory = GLOBAL; }
	| USE_KEYWORD DEST_KEYWORD GLOBAL_KEYWORD { dest_memory = GLOBAL; }
	| USE_KEYWORD CACHE_KEYWORD NONE_KEYWORD { cache_memory = NONE; }
	| USE_KEYWORD CACHE_KEYWORD LOCAL_KEYWORD { cache_memory = LOCAL; }
	| USE_KEYWORD CACHE_KEYWORD SHARED_KEYWORD { cache_memory = SHARED; }
	| SET_KEYWORD BLOCKS_KEYWORD INTEGER INTEGER { blocks = dim3($3,$4,1); }
	| SET_KEYWORD THREADS_KEYWORD INTEGER INTEGER { threads = dim3($3,$4,1); }
	| SET_KEYWORD TOLERANCE_KEYWORD DOUBLE { tolerance = double_stack[$3]; }
	| ALLOC_KEYWORD VARIABLE INTEGER INTEGER { 
			sym[$2] = (MATRIX<DATATYPE> *)malloc(sizeof(MATRIX<DATATYPE>)+($3 * $4 * sizeof(DATATYPE)));
			sym[$2]->height = $3;
			sym[$2]->width = $4;
		}
	| FREE_KEYWORD VARIABLE { free(sym[$2]); }
	| SHOW_KEYWORD INFO_KEYWORD { 
			int device_size = 0;
			cudaGetDeviceCount(&device_size);
			for (int i = 0; i < device_size; ++i)
			{
				cudaDeviceProp cudaDeviceProp;
				cudaGetDeviceProperties(&cudaDeviceProp, i);
				printf("Running on GPU %d (%s)\n", i, cudaDeviceProp.name); 

				printf("Device has ECC support enabled %d\n",cudaDeviceProp.ECCEnabled);
				printf("Number of asynchronous engines %d\n",cudaDeviceProp.asyncEngineCount);
				printf("Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer %d\n",cudaDeviceProp.canMapHostMemory);
				printf("Clock frequency in kilohertz %d\n",cudaDeviceProp.clockRate);
				printf("Compute mode (See cudaComputeMode) %d\n",cudaDeviceProp.computeMode);
				printf("Device can possibly execute multiple kernels concurrently %d\n",cudaDeviceProp.concurrentKernels);
				printf("Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. %d\n",cudaDeviceProp.deviceOverlap);
				printf("Device is integrated as opposed to discrete %d\n",cudaDeviceProp.integrated);
				printf("Specified whether there is a run time limit on kernels %d\n",cudaDeviceProp.kernelExecTimeoutEnabled);
				printf("Size of L2 cache in bytes %d\n",cudaDeviceProp.l2CacheSize);
				printf("Major compute capability %d\n",cudaDeviceProp.major);
				printf("Maximum size of each dimension of a grid %d\n",cudaDeviceProp.maxGridSize[0]);
				printf("Maximum size of each dimension of a grid %d\n",cudaDeviceProp.maxGridSize[1]);
				printf("Maximum size of each dimension of a grid %d\n",cudaDeviceProp.maxGridSize[2]);
				printf("Maximum 1D surface size %d\n",cudaDeviceProp.maxSurface1D);
				printf("Maximum 1D layered surface dimensions %d\n",cudaDeviceProp.maxSurface1DLayered[0]);
				printf("Maximum 1D layered surface dimensions %d\n",cudaDeviceProp.maxSurface1DLayered[1]);
				printf("Maximum 2D surface dimensions %d\n",cudaDeviceProp.maxSurface2D[0]);
				printf("Maximum 2D surface dimensions %d\n",cudaDeviceProp.maxSurface2D[1]);
				printf("Maximum 2D layered surface dimensions %d\n",cudaDeviceProp.maxSurface2DLayered[0]);
				printf("Maximum 2D layered surface dimensions %d\n",cudaDeviceProp.maxSurface2DLayered[1]);
				printf("Maximum 2D layered surface dimensions %d\n",cudaDeviceProp.maxSurface2DLayered[2]);
				printf("Maximum 3D surface dimensions %d\n",cudaDeviceProp.maxSurface3D[0]);
				printf("Maximum 3D surface dimensions %d\n",cudaDeviceProp.maxSurface3D[1]);
				printf("Maximum 3D surface dimensions %d\n",cudaDeviceProp.maxSurface3D[2]);
				printf("Maximum Cubemap surface dimensions %d\n",cudaDeviceProp.maxSurfaceCubemap);
				printf("Maximum Cubemap layered surface dimensions %d\n",cudaDeviceProp.maxSurfaceCubemapLayered[0]);
				printf("Maximum Cubemap layered surface dimensions %d\n",cudaDeviceProp.maxSurfaceCubemapLayered[1]);
				printf("Maximum 1D texture size %d\n",cudaDeviceProp.maxTexture1D);
				printf("Maximum 1D layered texture dimensions %d\n",cudaDeviceProp.maxTexture1DLayered[0]);
				printf("Maximum 1D layered texture dimensions %d\n",cudaDeviceProp.maxTexture1DLayered[1]);
				printf("Maximum size for 1D textures bound to linear memory %d\n",cudaDeviceProp.maxTexture1DLinear);
				printf("Maximum 1D mipmapped texture size %d\n",cudaDeviceProp.maxTexture1DMipmap);
				printf("Maximum 2D texture dimensions %d\n",cudaDeviceProp.maxTexture2D[0]);
				printf("Maximum 2D texture dimensions %d\n",cudaDeviceProp.maxTexture2D[1]);
				printf("Maximum 2D texture dimensions if texture gather operations have to be performed %d\n",cudaDeviceProp.maxTexture2DGather[0]);
				printf("Maximum 2D texture dimensions if texture gather operations have to be performed %d\n",cudaDeviceProp.maxTexture2DGather[1]);
				printf("Maximum 2D layered texture dimensions %d\n",cudaDeviceProp.maxTexture2DLayered[0]);
				printf("Maximum 2D layered texture dimensions %d\n",cudaDeviceProp.maxTexture2DLayered[1]);
				printf("Maximum 2D layered texture dimensions %d\n",cudaDeviceProp.maxTexture2DLayered[2]);
				printf("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory %d\n",cudaDeviceProp.maxTexture2DLinear[0]);
				printf("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory %d\n",cudaDeviceProp.maxTexture2DLinear[1]);
				printf("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory %d\n",cudaDeviceProp.maxTexture2DLinear[2]);
				printf("Maximum 2D mipmapped texture dimensions %d\n",cudaDeviceProp.maxTexture2DMipmap[0]);
				printf("Maximum 2D mipmapped texture dimensions %d\n",cudaDeviceProp.maxTexture2DMipmap[1]);
				printf("Maximum 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3D[0]);
				printf("Maximum 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3D[1]);
				printf("Maximum 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3D[2]);
				printf("Maximum alternate 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3DAlt[0]);
				printf("Maximum alternate 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3DAlt[1]);
				printf("Maximum alternate 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3DAlt[2]);
				printf("Maximum Cubemap texture dimensions %d\n",cudaDeviceProp.maxTextureCubemap);
				printf("Maximum Cubemap layered texture dimensions %d\n",cudaDeviceProp.maxTextureCubemapLayered[0]);
				printf("Maximum Cubemap layered texture dimensions %d\n",cudaDeviceProp.maxTextureCubemapLayered[1]);
				printf("Maximum size of each dimension of a block %d\n",cudaDeviceProp.maxThreadsDim[0]);
				printf("Maximum size of each dimension of a block %d\n",cudaDeviceProp.maxThreadsDim[1]);
				printf("Maximum size of each dimension of a block %d\n",cudaDeviceProp.maxThreadsDim[2]);
				printf("Maximum number of threads per block %d\n",cudaDeviceProp.maxThreadsPerBlock);
				printf("Maximum resident threads per multiprocessor %d\n",cudaDeviceProp.maxThreadsPerMultiProcessor);
				printf("Maximum pitch in bytes allowed by memory copies %d\n",cudaDeviceProp.memPitch);
				printf("Global memory bus width in bits %d\n",cudaDeviceProp.memoryBusWidth);
				printf("Peak memory clock frequency in kilohertz %d\n",cudaDeviceProp.memoryClockRate);
				printf("Minor compute capability %d\n",cudaDeviceProp.minor);
				printf("Number of multiprocessors on device %d\n",cudaDeviceProp.multiProcessorCount);
				printf("PCI bus ID of the device %d\n",cudaDeviceProp.pciBusID);
				printf("PCI device ID of the device %d\n",cudaDeviceProp.pciDeviceID);
				printf("PCI domain ID of the device %d\n",cudaDeviceProp.pciDomainID);
				printf("32-bit registers available per block %d\n",cudaDeviceProp.regsPerBlock);
				printf("Shared memory available per block in bytes %d\n",cudaDeviceProp.sharedMemPerBlock);
				printf("Device supports stream priorities %d\n",cudaDeviceProp.streamPrioritiesSupported);
				printf("Alignment requirements for surfaces %d\n",cudaDeviceProp.surfaceAlignment);
				printf("1 if device is a Tesla device using TCC driver, 0 otherwise %d\n",cudaDeviceProp.tccDriver);
				printf("Alignment requirement for textures %d\n",cudaDeviceProp.textureAlignment);
				printf("Pitch alignment requirement for texture references bound to pitched memory %d\n",cudaDeviceProp.texturePitchAlignment);
				printf("Constant memory available on device in bytes %d\n",cudaDeviceProp.totalConstMem);
				printf("Global memory available on device in bytes %d\n",cudaDeviceProp.totalGlobalMem);
				printf("Device shares a unified address space with the host %d\n",cudaDeviceProp.unifiedAddressing);
				printf("Warp size in threads %d\n",cudaDeviceProp.warpSize);

				fflush(stdout);
			}
		}
	| HELP_KEYWORD {
			printf("CUDA matrix calculator\n");
			printf("\thelp\n");
			printf("\tread|write VARIABLE FILENAME\n");
			printf("\tlet VARIABLE = VARIABLE [-+*] VARIABLE\n");
			printf("\tlet VARIABLE = [-+ inv rot gaussjordan] VARIABLE\n");
			printf("\tshow info\n");
			printf("\tuse [src dest cache] [texture constant global shared local none]\n");
			printf("\tset [blocks threads] INTEGER INTEGER\n");
			printf("\tset tolerance DOUBLE\n");
			printf("\talloc VARIABLE INTEGER INTEGER\n");
			printf("\tfree VARIABLE\n");
			fflush(stdout);
		}
	;
%%
void yyerror(char *s) {
	fprintf(stderr, "%s\n", s);
}
int main(void) {

	yyparse();

	cudaDeviceReset();

	while(string_stack_size-->0) free(string_stack[string_stack_size]);
}