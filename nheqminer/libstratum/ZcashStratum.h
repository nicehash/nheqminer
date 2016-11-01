#pragma once
// Copyright (c) 2016 Jack Grigg <jack@z.cash>
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "arith_uint256.h"
#include "primitives/block.h"
#include "uint256.h"
//#include "util.h"

#include <boost/signals2.hpp>
//#include <boost/signals.hpp>
//#include <boost/thread.hpp>
#include <thread>
#include <mutex>

#include "json/json_spirit_value.h"

#include "SolverStub.h"

#ifdef USE_CPU_TROMP
#include "../cpu_tromp/cpu_tromp.hpp"
#else
using cpu_tromp = SolverStub;
#endif
#ifdef USE_CPU_XENONCAT
#include "../cpu_xenoncat/cpu_xenoncat.hpp"
#else
using cpu_xenoncat = SolverStub1;
#endif
#ifdef USE_CUDA_TROMP
#include "../cuda_tromp/cuda_tromp.hpp"
#else
using cuda_tromp = SolverStub;
#endif
#ifdef USE_OCL_XMP
#include "../ocl_xpm/ocl_xmp.hpp"
#else
using ocl_xmp = SolverStub;
#endif


using namespace json_spirit;

extern int use_avx;
extern int use_avx2;

struct EquihashSolution
{
    uint256 nonce;
	std::string time;
	size_t nonce1size;
    std::vector<unsigned char> solution;

    EquihashSolution(uint256 n, std::vector<unsigned char> s, std::string t, size_t n1s)
		: nonce{ n }, solution{ s }, time{ t }, nonce1size{ n1s } { }

    std::string toString() const { return nonce.GetHex(); }
};

struct ZcashJob
{
    std::string job;
    CBlockHeader header;
    std::string time;
    size_t nonce1Size;
    arith_uint256 nonce2Space;
    arith_uint256 nonce2Inc;
    arith_uint256 serverTarget;
    bool clean;

    ZcashJob* clone() const;
    bool equals(const ZcashJob& a) const { return job == a.job; }

    // Access Stratum flags
    std::string jobId() const { return job; }
    bool cleanJobs() const { return clean; }

    void setTarget(std::string target);

    /**
     * Checks whether the given solution satisfies this work order.
     */
    bool evalSolution(const EquihashSolution* solution);

    /**
     * Returns a comma-separated string of Stratum submission values
     * corresponding to the given solution.
     */
    std::string getSubmission(const EquihashSolution* solution);
};

inline bool operator==(const ZcashJob& a, const ZcashJob& b)
{
    return a.equals(b);
}

typedef boost::signals2::signal<void (const ZcashJob*)> NewJob_t;

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
class ZcashMiner
{
    int nThreads;
	std::thread* minerThreads;
    //boost::thread_group* minerThreads;
    uint256 nonce1;
    size_t nonce1Size;
    arith_uint256 nonce2Space;
    arith_uint256 nonce2Inc;
    std::function<bool(const EquihashSolution&, const std::string&)> solutionFoundCallback;
	bool m_isActive;


	std::vector<CPUSolver*> cpu_contexts;
	std::vector<CUDASolver*> cuda_contexts;
	std::vector<OPENCLSolver*> opencl_contexts;


public:
    NewJob_t NewJob;
	bool* minerThreadActive;

	ZcashMiner(int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
		int opencl_count, int opencl_platf, int* opencl_en);
	~ZcashMiner();

    std::string userAgent();
    void start();
    void stop();
	bool isMining() { return m_isActive; }
	void setServerNonce(const std::string& n1str);
    ZcashJob* parseJob(const Array& params);
    void setJob(ZcashJob* job);
	void onSolutionFound(const std::function<bool(const EquihashSolution&, const std::string&)> callback);
	void submitSolution(const EquihashSolution& solution, const std::string& jobid);
    void acceptedSolution(bool stale);
    void rejectedSolution(bool stale);
    void failedSolution();

    static void doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
		int opencl_count, int opencl_platf, int* opencl_en);
};

typedef ZcashMiner<cpu_xenoncat, cuda_tromp, ocl_xmp> ZMinerAVX;
typedef ZcashMiner<cpu_tromp, cuda_tromp, ocl_xmp> ZMinerSSE2;

// gcc static undefined reference workaround
void ZMinerAVX_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
                           int opencl_count, int opencl_platf, int* opencl_en);
void ZMinerSSE2_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
                            int opencl_count, int opencl_platf, int* opencl_en);
