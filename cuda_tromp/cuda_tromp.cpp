#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>
#include <string>

#include "cuda_tromp.hpp"

struct proof;
#include "eqcuda.hpp"


SOLVER_NAME::SOLVER_NAME(int platf_id, int dev_id)
{
	device_id = dev_id;
	getinfo(0, dev_id, m_gpu_name, m_sm_count, m_version);

	// todo: determine default values for various GPUs here
	threadsperblock = 64;
	blocks = m_sm_count * 7;
}


std::string SOLVER_NAME::getdevinfo()
{
	return m_gpu_name + " (#" + std::to_string(device_id) + ") BLOCKS=" + 
		std::to_string(blocks) + ", THREADS=" + std::to_string(threadsperblock);
}


int SOLVER_NAME::getcount()
{
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	return device_count;
}

void SOLVER_NAME::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)
{
	//int runtime_version;
	//checkCudaErrors(cudaRuntimeGetVersion(&runtime_version));

	cudaDeviceProp device_props;

	checkCudaErrors(cudaGetDeviceProperties(&device_props, d_id));

	gpu_name = device_props.name;
	sm_count = device_props.multiProcessorCount;
	version = std::to_string(device_props.major) + "." + std::to_string(device_props.minor);
}


void SOLVER_NAME::start(SOLVER_NAME& device_context)
{ 
	device_context.context = new eq_cuda_context(device_context.threadsperblock, 
		device_context.blocks,
		device_context.device_id);
}

void SOLVER_NAME::stop(SOLVER_NAME& device_context)
{ 
	if (device_context.context)
		delete device_context.context;
}

void SOLVER_NAME::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef,
	SOLVER_NAME& device_context)
{
	device_context.context->solve(tequihash_header,
		tequihash_header_len,
		nonce,
		nonce_len,
		cancelf,
		solutionf,
		hashdonef);
}