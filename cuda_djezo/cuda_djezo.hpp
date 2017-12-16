#pragma once

#ifdef _LIB
#define DLL_CUDA_DJEZO __declspec(dllexport)
#else
#define DLL_CUDA_DJEZO
#endif

struct eq_cuda_context_interface;

struct DLL_CUDA_DJEZO cuda_djezo
{
	int threadsperblock;
	int blocks;
	int device_id;
	int combo_mode;
	eq_cuda_context_interface* context;

	cuda_djezo(int platf_id, int dev_id);

	std::string getdevinfo();

	static int getcount();

	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

	static void start(cuda_djezo& device_context);

	static void stop(cuda_djezo& device_context);

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		cuda_djezo& device_context);

	std::string getname() { return "CUDA-DJEZO"; }

private:
	std::string m_gpu_name;
	std::string m_version;
	int m_sm_count;
};