#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions_decls.h"
#include "../cpu_tromp/blake2/blake2.h"
#include "cuda_djezo.hpp"

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

#define checkCudaDriverErrors(call)								\
do {														\
	CUresult err = call;									\
	if (CUDA_SUCCESS != err) {								\
		char errorBuff[512];								\
		_SNPRINTF(errorBuff, sizeof(errorBuff) - 1,			\
			"CUDA error DRIVER: '%d' in func '%s' line %d",			\
			err, __FUNCTION__, __LINE__);	\
		throw std::runtime_error(errorBuff);				\
				}														\
} while (0)

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef unsigned char uchar;

struct packer_default;
struct packer_cantor;

#define MAXREALSOLS 9

struct scontainerreal
{
	u32 sols[MAXREALSOLS][512];
	u32 nsols;
};

template <u32 RB, u32 SM>
struct equi;

struct eq_cuda_context_interface
{
	virtual ~eq_cuda_context_interface();

	virtual void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);
};


template <u32 RB, u32 SM, u32 SSM, u32 THREADS, typename PACKER>
struct eq_cuda_context : public eq_cuda_context_interface
{
	int threadsperblock;
	int totalblocks;
	int device_id;
	equi<RB, SM>* device_eq;
	scontainerreal* solutions;
	CUcontext pctx;

	eq_cuda_context(int id);
	~eq_cuda_context();

	void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef);
};

#define CONFIG_MODE_1	9, 1248, 12, 640, packer_cantor

#define CONFIG_MODE_2	8, 640, 12, 512, packer_default