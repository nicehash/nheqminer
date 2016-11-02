#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <string.h>
#include <stdlib.h>
#endif

#include "cpu_xenoncat.hpp"

#define CONTEXT_SIZE 178033152

//#define USE_XENON_DLL

#ifdef __cplusplus
extern "C" 
#endif
{
#ifndef USE_XENON_DLL
	//Linkage with assembly
	//EhPrepare takes in 136 bytes of input. The remaining 4 bytes of input is fed as nonce to EhSolver.
	//EhPrepare saves the 136 bytes in context, and EhSolver can be called repeatedly with different nonce.
	void EhPrepareAVX1(void *context, void *input);
	int32_t EhSolverAVX1(void *context, uint32_t nonce);

	void EhPrepareAVX2(void *context, void *input);
	int32_t EhSolverAVX2(void *context, uint32_t nonce);

#else
	typedef void(__fastcall *_EhPrepare)(void*, void*);
	_EhPrepare EhPrepare;

	typedef int32_t(__fastcall *_EhSolver)(void*, uint32_t);
	_EhSolver EhSolver;

	void init_library(int use_avx2)
	{
		HMODULE hmod;
		if (use_avx2) hmod = LoadLibraryA("xenoncat_AVX2.dll");
		else hmod = LoadLibraryA("xenoncat_AVX.dll");
		EhPrepare = (_EhPrepare)GetProcAddress(hmod, "EhPrepare");
		EhSolver = (_EhSolver)GetProcAddress(hmod, "EhSolver");

		if (!EhPrepare || !EhSolver)
		{
			puts("Library xenoncat is missing.");
			exit(0);
		}
	}
#endif

#ifdef __cplusplus
}
#endif

void cpu_xenoncat::start(cpu_xenoncat& device_context) 
{
#ifdef USE_XENON_DLL
	init_library(device_context.use_opt);
#endif
	device_context.memory_alloc = malloc(CONTEXT_SIZE + 4096);
	device_context.memory = (void*)(((long long)device_context.memory_alloc + 4095) & -4096);

	// todo: improve memory; LOCKED_PAGES ?
}

void cpu_xenoncat::stop(cpu_xenoncat& device_context) 
{ 
	free(device_context.memory_alloc);
}

void cpu_xenoncat::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef,
	cpu_xenoncat& device_context)
{
	unsigned char context[140];
	int32_t i, numsolutions;

	memcpy(context, tequihash_header, 108);
	memcpy(context + 108, nonce, 32);

#ifdef USE_XENON_DLL
	EhPrepare(device_context.memory, (void *)context);
	numsolutions = EhSolver(device_context.memory, *(uint32_t *)(context + 136));
#else
	if (device_context.use_opt)
	{
		EhPrepareAVX2(device_context.memory, (void *)context);
		numsolutions = EhSolverAVX2(device_context.memory, *(uint32_t *)(context + 136));
	}
	else
	{
		EhPrepareAVX1(device_context.memory, (void *)context);
		numsolutions = EhSolverAVX1(device_context.memory, *(uint32_t *)(context + 136));
	}
#endif
	for (i = 0; i < numsolutions; i++) 
	{
		//printf("Solution found, start: %08x\n", *(uint32_t*)((unsigned char*)device_context.memory + (1344 * i)));
		solutionf(std::vector<uint32_t>(0), 1344, (unsigned char*)device_context.memory + (1344 * i));
		if (cancelf()) return;
		//validBlock(validBlockData, (unsigned char*)context + (1344 * i));
	}
	hashdonef();
}