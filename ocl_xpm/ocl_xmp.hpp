#pragma once
#ifdef _LIB
#define DLL_OCL_XMP __declspec(dllexport)
#else
#define DLL_OCL_XMP
#endif

// remove after
#include <string>
#include <functional>
#include <vector>
#include <cstdint>

struct MinerInstance;

struct DLL_OCL_XMP ocl_xmp
{
	//int threadsperblock;
	int blocks;
	int device_id;
	int platform_id;

	MinerInstance* context;
	// threads
	unsigned threadsNum; // TMP
	unsigned wokrsize;

	bool is_init_success = false;

	ocl_xmp(int platf_id, int dev_id);

	std::string getdevinfo();

	static int getcount();

	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

	static void start(ocl_xmp& device_context);

	static void stop(ocl_xmp& device_context);

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		ocl_xmp& device_context);

	std::string getname() { return "OCL_XMP"; }

private:
	std::string m_gpu_name;
	std::string m_version;
};