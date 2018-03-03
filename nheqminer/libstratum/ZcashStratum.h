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

#include "ISolver.h"

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
		: nonce{ n }, nonce1size{ n1s } { solution = s; time = t; }

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

	std::vector<ISolver *> solvers;

public:
    NewJob_t NewJob;
	bool* minerThreadActive;

	ZcashMiner(const std::vector<ISolver *> &i_solvers);
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
};

void Solvers_doBenchmark(int hashes, const std::vector<ISolver *> &solvers);

