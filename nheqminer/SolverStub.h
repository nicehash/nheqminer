#pragma once

#include <string>
#include <functional>
#include <vector>
#include <cstdint>

#define CREATE_SOLVER_STUB(NAME, STUB_NAME) \
struct NAME { \
    int threadsperblock; \
    int blocks; \
    int use_opt; \
    NAME() {} \
    NAME(int platf_id, int dev_id) {} \
    std::string getdevinfo() { return ""; } \
    static int getcount() { return 0; } \
    static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version)  {} \
    static void start(NAME& device_context)  {} \
    static void stop(NAME& device_context)  {} \
    static void solve(const char *tequihash_header, \
        unsigned int tequihash_header_len, \
        const char* nonce, \
        unsigned int nonce_len, \
        std::function<bool()> cancelf, \
        std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf, \
        std::function<void(void)> hashdonef, \
        NAME& device_context)  {} \
    std::string getname() { return STUB_NAME; } \
    static void print_opencl_devices()  {} \
};
