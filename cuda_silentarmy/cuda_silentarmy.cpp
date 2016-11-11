#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>
#include <string>

#include "cuda_silentarmy.hpp"
#include "sa_cuda_context.hpp"

cuda_sa_solver::cuda_sa_solver(int platf_id, int dev_id)
{
	device_id = dev_id;
	getinfo(0, dev_id, m_gpu_name, m_sm_count, m_version);

	// todo: determine default values for various GPUs here
	threadsperblock = 64;
	blocks = m_sm_count * 7;
}

std::string cuda_sa_solver::getdevinfo()
{
	return m_gpu_name + " (#" + std::to_string(device_id) + ") BLOCKS=" +
		std::to_string(blocks) + ", THREADS=" + std::to_string(threadsperblock);
}

int cuda_sa_solver::getcount()
{
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	return device_count;
}

void cuda_sa_solver::getinfo(int platf_id, int d_id, std::string & gpu_name, int & sm_count, std::string & version)
{
	cudaDeviceProp device_props;

	checkCudaErrors(cudaGetDeviceProperties(&device_props, d_id));

	gpu_name = device_props.name;
	sm_count = device_props.multiProcessorCount;
	version = std::to_string(device_props.major) + "." + std::to_string(device_props.minor);

}

void cuda_sa_solver::start(cuda_sa_solver & device_context)
{
	device_context.context = new sa_cuda_context(device_context.threadsperblock,
		device_context.blocks,
		device_context.device_id);

}

void cuda_sa_solver::stop(cuda_sa_solver & device_context)
{
	if (device_context.context)
		delete device_context.context;
}

void cuda_sa_solver::solve(const char * tequihash_header, unsigned int tequihash_header_len, const char * nonce, unsigned int nonce_len, std::function<bool()> cancelf, std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf, std::function<void(void)> hashdonef, cuda_sa_solver & device_context)
{
	device_context.context->solve(tequihash_header,
		tequihash_header_len,
		nonce,
		nonce_len,
		cancelf,
		solutionf,
		hashdonef);
}
