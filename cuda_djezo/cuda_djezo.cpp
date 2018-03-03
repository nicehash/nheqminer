#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>
#include <string>

#include "cuda_djezo.hpp"

struct proof;
#include "eqcuda.hpp"


cuda_djezo::cuda_djezo(int platf_id, int dev_id)
{
	device_id = dev_id;
	getinfo(0, dev_id, m_gpu_name, m_sm_count, m_version);

	combo_mode = 1;

	int major, minor;
	std::string::size_type n = m_version.find(".");
	if (n != std::string::npos)
	{
		major = atoi(m_version.substr(0, n).c_str());
		minor = atoi(m_version.substr(n + 1, m_version.length() - n - 1).c_str());

		if (major < 5)
		{
			throw std::runtime_error("Only CUDA devices with SM 5.0 and higher are supported.");
		}
		else if (major == 5 && minor == 0)
		{
			combo_mode = 2;
		}
	}
	else
		throw std::runtime_error("Uknown Compute/SM version.");
}


std::string cuda_djezo::getdevinfo()
{
	return m_gpu_name + " (#" + std::to_string(device_id) + ") M=" + std::to_string(combo_mode);
}


int cuda_djezo::getcount()
{
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	return device_count;
}

void cuda_djezo::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)
{
	//int runtime_version;
	//checkCudaErrors(cudaRuntimeGetVersion(&runtime_version));

	cudaDeviceProp device_props;

	checkCudaErrors(cudaGetDeviceProperties(&device_props, d_id));

	gpu_name = device_props.name;
	sm_count = device_props.multiProcessorCount;
	version = std::to_string(device_props.major) + "." + std::to_string(device_props.minor);
}


void cuda_djezo::start(cuda_djezo& device_context)
{ 
	switch (device_context.combo_mode)
	{
#ifdef CONFIG_MODE_2
	case 2:
		device_context.context = new eq_cuda_context<CONFIG_MODE_2>(device_context.device_id);
		break;
#endif
#ifdef CONFIG_MODE_3
	case 3:
		device_context.context = new eq_cuda_context<CONFIG_MODE_3>(device_context.device_id);
		break;
#endif
	default:
		device_context.context = new eq_cuda_context<CONFIG_MODE_1>(device_context.device_id);
		break;
	}
}

void cuda_djezo::stop(cuda_djezo& device_context)
{ 
	if (device_context.context)
	{
		delete device_context.context;
		device_context.context = nullptr;
	}
}

void cuda_djezo::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef,
	cuda_djezo& device_context)
{
	device_context.context->solve(tequihash_header,
		tequihash_header_len,
		nonce,
		nonce_len,
		cancelf,
		solutionf,
		hashdonef);
}


void eq_cuda_context_interface::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef)
{
}


eq_cuda_context_interface::~eq_cuda_context_interface() { }