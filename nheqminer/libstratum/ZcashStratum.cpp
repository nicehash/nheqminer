// Copyright (c) 2016 Jack Grigg <jack@z.cash>
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "ZcashStratum.h"

//#include "chainparams.h"
#include "crypto/equihash.h"
#include "streams.h"
#include "version.h"

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <boost/thread/exceptions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/circular_buffer.hpp>
#include "speed.hpp"

#ifdef WIN32
#include <Windows.h>
#endif


#define BOOST_LOG_CUSTOM(sev, pos) BOOST_LOG_TRIVIAL(sev) << "miner#" << pos << " | "


void static ZcashMinerThread(ZcashMiner* miner, int size, int pos)
{
	BOOST_LOG_CUSTOM(info, pos) << "Starting thread";

	unsigned int n = PARAMETER_N;
	unsigned int k = PARAMETER_K;

    std::shared_ptr<std::mutex> m_zmt(new std::mutex);
    CBlockHeader header;
    arith_uint256 space;
    size_t offset;
    arith_uint256 inc;
    arith_uint256 target;
    std::atomic_bool workReady {false};
    std::atomic_bool cancelSolver {false};

    miner->NewJob.connect(NewJob_t::slot_type(
        [&m_zmt, &header, &space, &offset, &inc, &target, &workReady, &cancelSolver]
        (const ZcashJob* job) mutable {
            std::lock_guard<std::mutex> lock{*m_zmt.get()};
            if (job) {
                header = job->header;
                space = job->nonce2Space;
                offset = job->nonce1Size * 4; // Hex length to bit length
                inc = job->nonce2Inc;
                target = job->serverTarget;
                workReady.store(true);
                if (job->clean) {
                    cancelSolver.store(true);
                }
            } else {
                workReady.store(false);
                cancelSolver.store(true);
            }
        }
    ).track_foreign(m_zmt)); // So the signal disconnects when the mining thread exits

    try {
        while (true) {
            // Wait for work
            bool expected;
            do {
                expected = true;
				if (!miner->minerThreadActive[pos])
					throw boost::thread_interrupted();
                //boost::this_thread::interruption_point();
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            } while (!workReady.compare_exchange_weak(expected, false));
            // TODO change atomically with workReady
            cancelSolver.store(false);

            // Calculate nonce limits
            arith_uint256 nonce;
            arith_uint256 nonceEnd;
            {
                std::lock_guard<std::mutex> lock{*m_zmt.get()};
                arith_uint256 baseNonce = UintToArith256(header.nNonce);
                nonce = baseNonce + ((space/size)*pos << offset);
                nonceEnd = baseNonce + ((space/size)*(pos+1) << offset);
            }

            // Hash state
            blake2b_state state;
            EhInitialiseState(n, k, state);

            // I = the block header minus nonce and solution.
            CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
            {
                std::lock_guard<std::mutex> lock{*m_zmt.get()};
                CEquihashInput I{header};
                ss << I;
            }

            // H(I||...
            blake2b_update(&state, (unsigned char*)&ss[0], ss.size());

            // Start working
            while (true) {
                // H(I||V||...
                blake2b_state curr_state;
                curr_state = state;
				//nonce = arith_uint256("1e11003000000000000000003000111ffcffffffffffffffffffffffffffff3f");
				//nonce = arith_uint256("3ffffffffffffffffffffffffffffffc1f11003000000000000000003000111e");
				auto bNonce = ArithToUint256(nonce);
                blake2b_update(&curr_state,
                        bNonce.begin(),
                        bNonce.size());

                // (x_1, x_2, ...) = A(I, V, n, k)
				BOOST_LOG_CUSTOM(debug, pos) << "Running Equihash solver with nNonce = " << nonce.ToString();

                std::function<bool(std::vector<unsigned char>)> validBlock =
                        [&m_zmt, &header, &bNonce, &target, &miner, pos]
                        (std::vector<unsigned char> soln) {
                    std::lock_guard<std::mutex> lock{*m_zmt.get()};
                    // Write the solution to the hash and compute the result.
					BOOST_LOG_CUSTOM(debug, pos) << "Checking solution against target...";
                    header.nNonce = bNonce;
                    header.nSolution = soln;

					speed.AddSolution();

					uint256 headerhash = header.GetHash();
					if (UintToArith256(headerhash) > target) {
						BOOST_LOG_CUSTOM(debug, pos) << "Too large: " << headerhash.ToString();
                        return false;
                    }

                    // Found a solution
					BOOST_LOG_CUSTOM(debug, pos) << "Found solution with header hash: " << headerhash.ToString();
                    EquihashSolution solution {bNonce, soln};
                    miner->submitSolution(solution);

                    // We're a pooled miner, so try all solutions
                    return false;
                };

                std::function<bool(EhSolverCancelCheck)> cancelled =
                        [&cancelSolver, &miner, pos](EhSolverCancelCheck pos1) 
				{
					if (!miner->minerThreadActive[pos])
						throw boost::thread_interrupted();
                    //boost::this_thread::interruption_point();
                    return cancelSolver.load();
                };

                try 
				{
                    // If we find a valid block, we get more work
                    if (EhOptimisedSolve(n, k, curr_state, validBlock, cancelled))
					{
						speed.AddHash(); // found block, hash was done
                        break;
                    }
					speed.AddHash(); // did not cancel in the middle, hash was done
                } 
				catch (EhSolverCancelledException&) 
				{
					speed.AddHashInterrupted();
					BOOST_LOG_CUSTOM(debug, pos) << "Equihash solver cancelled";
                    cancelSolver.store(false);
                    break;
                }

                // Check for stop
				if (!miner->minerThreadActive[pos])
					throw boost::thread_interrupted();
                //boost::this_thread::interruption_point();
                if (nonce == nonceEnd) {
                    break;
                }

                // Check for new work
                if (workReady.load()) {
					BOOST_LOG_CUSTOM(debug, pos) << "New work received, dropping current work";
                    break;
                }

                // Update nonce
                nonce += inc;
            }
        }
    }
    catch (const boost::thread_interrupted&)
    {
		BOOST_LOG_CUSTOM(info, pos) << "Thread #" << pos << " terminated";
        //throw;
		return;
    }
    catch (const std::runtime_error &e)
    {
		BOOST_LOG_CUSTOM(info, pos) << "Runtime error: " << e.what();
        return;
    }
}

ZcashJob* ZcashJob::clone() const
{
    ZcashJob* ret = new ZcashJob();
    ret->job = job;
    ret->header = header;
    ret->time = time;
    ret->nonce1Size = nonce1Size;
    ret->nonce2Space = nonce2Space;
    ret->nonce2Inc = nonce2Inc;
    ret->serverTarget = serverTarget;
    ret->clean = clean;
    return ret;
}

void ZcashJob::setTarget(std::string target)
{
    if (target.size() > 0) {
        serverTarget = UintToArith256(uint256S(target));
    } else {
		BOOST_LOG_TRIVIAL(debug) << "miner | New job but no server target, assuming powLimit";
		serverTarget = UintToArith256(uint256S("0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"));
    }
}

bool ZcashJob::evalSolution(const EquihashSolution* solution)
{
	unsigned int n = PARAMETER_N;
	unsigned int k = PARAMETER_K;

    // Hash state
    blake2b_state state;
    EhInitialiseState(n, k, state);

    // I = the block header minus nonce and solution.
    CEquihashInput I{header};
    // I||V
    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
    ss << I;
    ss << solution->nonce;

    // H(I||V||...
    blake2b_update(&state, (unsigned char*)&ss[0], ss.size());

    bool isValid;
    EhIsValidSolution(n, k, state, solution->solution, isValid);
    return isValid;
}

std::string ZcashJob::getSubmission(const EquihashSolution* solution)
{
    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
    ss << solution->nonce;
    ss << solution->solution;
    std::string strHex = HexStr(ss.begin(), ss.end());

    std::stringstream stream;
    stream << "\"" << job;
    stream << "\",\"" << time;
    stream << "\",\"" << strHex.substr(nonce1Size, 64-nonce1Size);
    stream << "\",\"" << strHex.substr(64);
    stream << "\"";
    return stream.str();
}

ZcashMiner::ZcashMiner(int threads)
    : nThreads{threads}, minerThreads{nullptr}
{
	m_isActive = false;
    if (nThreads < 1) {
		nThreads = std::thread::hardware_concurrency();
    }
}

std::string ZcashMiner::userAgent()
{
	return "equihashminer/0." STANDALONE_MINER_VERSION;
}

void ZcashMiner::start()
{
    if (minerThreads) {
        stop();
    }

	speed.Reset();

	m_isActive = true;

	minerThreads = new std::thread[nThreads];
	minerThreadActive = new bool[nThreads];
	for (int i = 0; i < nThreads; i++) 
	{
		minerThreadActive[i] = true;
		minerThreads[i] = std::thread(boost::bind(&ZcashMinerThread, this, nThreads, i));
#ifdef WIN32
		HANDLE hThread = minerThreads[i].native_handle();
		if (!SetThreadPriority(hThread, THREAD_PRIORITY_LOWEST))
		{
			BOOST_LOG_CUSTOM(warning, i) << "Failed to set low priority";
		}
		else
		{
			BOOST_LOG_CUSTOM(debug, i) << "Priority set to " << GetThreadPriority(hThread);
		}
#else
		// todo: linux set low priority
#endif
	}
    /*minerThreads = new boost::thread_group();
    for (int i = 0; i < nThreads; i++) {
        minerThreads->create_thread(boost::bind(&ZcashMinerThread, this, nThreads, i));
    }*/
}

void ZcashMiner::stop()
{
	m_isActive = false;
	if (minerThreads)
	{
		for (int i = 0; i < nThreads; i++)
			minerThreadActive[i] = false;
		for (int i = 0; i < nThreads; i++)
			minerThreads[i].join();
		delete minerThreads;
		delete minerThreadActive;
	}
    /*if (minerThreads) {
        minerThreads->interrupt_all();
        delete minerThreads;
        minerThreads = nullptr;
    }*/
}

//void ZcashMiner::setServerNonce(const Array& params)
void ZcashMiner::setServerNonce(const std::string& n1str)
{
    //auto n1str = params[1].get_str();
	BOOST_LOG_TRIVIAL(info) << "miner | Extranonce is " << n1str;
    std::vector<unsigned char> nonceData(ParseHex(n1str));
    while (nonceData.size() < 32) {
        nonceData.push_back(0);
    }
    CDataStream ss(nonceData, SER_NETWORK, PROTOCOL_VERSION);
    ss >> nonce1;

    nonce1Size = n1str.size();
    size_t nonce1Bits = nonce1Size * 4; // Hex length to bit length
    size_t nonce2Bits = 256 - nonce1Bits;

    nonce2Space = 1;
    nonce2Space <<= nonce2Bits;
    nonce2Space -= 1;

    nonce2Inc = 1;
    nonce2Inc <<= nonce1Bits;
}

ZcashJob* ZcashMiner::parseJob(const Array& params)
{
    if (params.size() < 2) {
        throw std::logic_error("Invalid job params");
    }

    ZcashJob* ret = new ZcashJob();
    ret->job = params[0].get_str();

    int32_t version;
    sscanf(params[1].get_str().c_str(), "%x", &version);
    // TODO: On a LE host shouldn't this be le32toh?
    ret->header.nVersion = be32toh(version);

    if (ret->header.nVersion == 4) {
        if (params.size() < 8) {
            throw std::logic_error("Invalid job params");
        }

        std::stringstream ssHeader;
        ssHeader << params[1].get_str()
                 << params[2].get_str()
                 << params[3].get_str()
                 << params[4].get_str()
                 << params[5].get_str()
                 << params[6].get_str()
                    // Empty nonce
                 << "0000000000000000000000000000000000000000000000000000000000000000"
                 << "00"; // Empty solution
        auto strHexHeader = ssHeader.str();
        std::vector<unsigned char> headerData(ParseHex(strHexHeader));
        CDataStream ss(headerData, SER_NETWORK, PROTOCOL_VERSION);
        try {
            ss >> ret->header;
        } catch (const std::ios_base::failure&) {
            throw std::logic_error("ZcashMiner::parseJob(): Invalid block header parameters");
        }

        ret->time = params[5].get_str();
        ret->clean = params[7].get_bool();
    } else {
        throw std::logic_error("ZcashMiner::parseJob(): Invalid or unsupported block header version");
    }

    ret->header.nNonce = nonce1;
    ret->nonce1Size = nonce1Size;
    ret->nonce2Space = nonce2Space;
    ret->nonce2Inc = nonce2Inc;

    return ret;
}

void ZcashMiner::setJob(ZcashJob* job)
{
    NewJob(job);
}

void ZcashMiner::onSolutionFound(
        const std::function<bool(const EquihashSolution&)> callback)
{
    solutionFoundCallback = callback;
}

void ZcashMiner::submitSolution(const EquihashSolution& solution)
{
    solutionFoundCallback(solution);
	speed.AddShare();
}

void ZcashMiner::acceptedSolution(bool stale)
{
	speed.AddShareOK();
}

void ZcashMiner::rejectedSolution(bool stale)
{
}

void ZcashMiner::failedSolution()
{
}
