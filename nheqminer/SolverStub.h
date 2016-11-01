#pragma once

#include <string>
#include <functional>
#include <vector>
#include <cstdint>


struct SolverStub {

	int threadsperblock;
	int blocks;

	int use_opt;

	SolverStub() {}
	SolverStub(int platf_id, int dev_id) {}

	std::string getdevinfo() { return ""; }

	static int getcount() { return 0; }

	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)  {}

	static void start(SolverStub& device_context)  {}

	static void stop(SolverStub& device_context)  {}

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		SolverStub& device_context)  {}

	std::string getname() { return "SolverStub"; }

	static void print_opencl_devices()  {}
};
// fix this workaround later
struct SolverStub1 {

    int threadsperblock;
    int blocks;

    int use_opt;

    SolverStub1() {}
    SolverStub1(int platf_id, int dev_id) {}

    std::string getdevinfo() { return ""; }

    static int getcount() { return 0; }

    static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)  {}

    static void start(SolverStub1& device_context)  {}

    static void stop(SolverStub1& device_context)  {}

    static void solve(const char *tequihash_header,
        unsigned int tequihash_header_len,
        const char* nonce,
        unsigned int nonce_len,
        std::function<bool()> cancelf,
        std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
        std::function<void(void)> hashdonef,
        SolverStub1& device_context)  {}

    std::string getname() { return "SolverStub1"; }

    static void print_opencl_devices()  {}
};
