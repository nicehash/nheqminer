#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef WIN32
#define _SNPRINTF _snprintf
#else
#include <stdio.h>
#define _SNPRINTF snprintf
#endif

#define checkCudaErrors(call)								\
do {														\
	cudaError_t err = call;									\
	if (cudaSuccess != err) {								\
		char errorBuff[512];								\
        _SNPRINTF(errorBuff, sizeof(errorBuff) - 1,			\
			"CUDA error '%s' in func '%s' line %d",			\
			cudaGetErrorString(err), __FUNCTION__, __LINE__);	\
		throw std::runtime_error(errorBuff);				\
		}														\
} while (0)

struct equi;

struct eq_cuda_context
{
	int threadsperblock;
	int totalblocks;
	int device_id;
	equi* eq;
	equi* device_eq;
	uint32_t *heap0, *heap1;
	void* sol_memory;
	proof* solutions;

	eq_cuda_context(int tpb, int blocks, int id);
	~eq_cuda_context();

	void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);
};
