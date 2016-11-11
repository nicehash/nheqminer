#pragma once

#ifdef _LIB
#define DLL_CUDA_SA __declspec(dllexport)
#else
#define DLL_CUDA_SA
#endif

struct sa_cuda_context;

struct DLL_CUDA_SA cuda_sa_solver
{
	int threadsperblock;
	int blocks;
	int device_id;
	sa_cuda_context* context;

	cuda_sa_solver(int platf_id, int dev_id);

	std::string getdevinfo();

	static int getcount();

	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

	static void start(cuda_sa_solver& device_context);

	static void stop(cuda_sa_solver& device_context);

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		cuda_sa_solver& device_context);

	std::string getname() { return "CUDA-SILENTARMY"; }

private:
	std::string m_gpu_name;
	std::string m_version;
	int m_sm_count;
};
