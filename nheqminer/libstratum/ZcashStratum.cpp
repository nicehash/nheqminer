// Copyright (c) 2016 Jack Grigg <jack@z.cash>
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "version.h"
#include "ZcashStratum.h"

#include "utilstrencodings.h"
#include "trompequihash/equi_miner.h"
#include "streams.h"

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <inttypes.h>
#include <boost/thread/exceptions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/circular_buffer.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

#include "speed.hpp"

#ifdef WIN32
#include <Windows.h>
#endif

#include <boost/static_assert.hpp>

#if defined(__clang__)
// support for clang and AVX1/2 detection
#include <cpuid.h>
#endif

extern int equiengine;
typedef uint32_t eh_index;


#define BOOST_LOG_CUSTOM(sev, pos) BOOST_LOG_TRIVIAL(sev) << "miner#" << pos << " | "

#ifdef XENONCAT
#define CONTEXT_SIZE 178033152

#ifdef __MINGW32__
extern "C" void __attribute__((sysv_abi)) EhPrepareAVX1(void *context, void *input);
extern "C" int32_t __attribute__((sysv_abi)) EhSolverAVX1(void *context, uint32_t nonce);
extern "C" void __attribute__((sysv_abi)) EhPrepareAVX2(void *context, void *input);
extern "C" int32_t __attribute__((sysv_abi)) EhSolverAVX2(void *context, uint32_t nonce);
void __attribute__((sysv_abi)) (*EhPrepare)(void *,void *);
int32_t __attribute__((sysv_abi)) (*EhSolver)(void *, uint32_t);
#else
extern "C" void EhPrepareAVX1(void *context, void *input);
extern "C" int32_t EhSolverAVX1(void *context, uint32_t nonce);
extern "C" void EhPrepareAVX2(void *context, void *input);
extern "C" int32_t EhSolverAVX2(void *context, uint32_t nonce);
void (*EhPrepare)(void *,void *);
int32_t (*EhSolver)(void *, uint32_t);
#endif
#endif

void CompressArray(const unsigned char* in, size_t in_len,
	unsigned char* out, size_t out_len,
	size_t bit_len, size_t byte_pad)
{
	assert(bit_len >= 8);
	assert(8 * sizeof(uint32_t) >= 7 + bit_len);

	size_t in_width{ (bit_len + 7) / 8 + byte_pad };
	assert(out_len == bit_len*in_len / (8 * in_width));

	uint32_t bit_len_mask{ ((uint32_t)1 << bit_len) - 1 };

	// The acc_bits least-significant bits of acc_value represent a bit sequence
	// in big-endian order.
	size_t acc_bits = 0;
	uint32_t acc_value = 0;

	size_t j = 0;
	for (size_t i = 0; i < out_len; i++) {
		// When we have fewer than 8 bits left in the accumulator, read the next
		// input element.
		if (acc_bits < 8) {
			acc_value = acc_value << bit_len;
			for (size_t x = byte_pad; x < in_width; x++) {
				acc_value = acc_value | (
					(
					// Apply bit_len_mask across byte boundaries
					in[j + x] & ((bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF)
					) << (8 * (in_width - x - 1))); // Big-endian
			}
			j += in_width;
			acc_bits += bit_len;
		}

		acc_bits -= 8;
		out[i] = (acc_value >> acc_bits) & 0xFF;
	}
}


void EhIndexToArray(const eh_index i, unsigned char* array)
{
	BOOST_STATIC_ASSERT(sizeof(eh_index) == 4);
	eh_index bei = htobe32(i);
	memcpy(array, &bei, sizeof(eh_index));
}


std::vector<unsigned char> GetMinimalFromIndices(std::vector<eh_index> indices,
	size_t cBitLen)
{
	assert(((cBitLen + 1) + 7) / 8 <= sizeof(eh_index));
	size_t lenIndices{ indices.size()*sizeof(eh_index) };
	size_t minLen{ (cBitLen + 1)*lenIndices / (8 * sizeof(eh_index)) };
	size_t bytePad{ sizeof(eh_index) - ((cBitLen + 1) + 7) / 8 };
	std::vector<unsigned char> array(lenIndices);
	for (int i = 0; i < indices.size(); i++) {
		EhIndexToArray(indices[i], array.data() + (i*sizeof(eh_index)));
	}
	std::vector<unsigned char> ret(minLen);
	CompressArray(array.data(), lenIndices,
		ret.data(), minLen, cBitLen + 1, bytePad);
	return ret;
}

#ifdef XENONCAT
void static XenoncatZcashMinerThread(ZcashMiner* miner, int size, int pos)
{
	BOOST_LOG_CUSTOM(info, pos) << "Starting thread #" << pos;

	unsigned int n = PARAMETER_N;
	unsigned int k = PARAMETER_K;

    std::shared_ptr<std::mutex> m_zmt(new std::mutex);
    CBlockHeader header;
    arith_uint256 space;
    size_t offset;
    arith_uint256 inc;
    arith_uint256 target;
	std::string jobId;
	std::string nTime;
    std::atomic_bool workReady {false};
    std::atomic_bool cancelSolver {false};
	std::atomic_bool pauseMining {false};

    miner->NewJob.connect(NewJob_t::slot_type(
		[&m_zmt, &header, &space, &offset, &inc, &target, &workReady, &cancelSolver, pos, &pauseMining, &jobId, &nTime]
	(const ZcashJob* job) mutable {
	    std::lock_guard<std::mutex> lock{*m_zmt.get()};
	    if (job) {
				BOOST_LOG_CUSTOM(debug, pos) << "Loading new job #" << job->jobId();
				jobId = job->jobId();
				nTime = job->time;
		header = job->header;
		space = job->nonce2Space;
		offset = job->nonce1Size * 4; // Hex length to bit length
		inc = job->nonce2Inc;
		target = job->serverTarget;
				pauseMining.store(false);
		workReady.store(true);
		/*if (job->clean) {
		    cancelSolver.store(true);
		}*/
	    } else {
		workReady.store(false);
		cancelSolver.store(true);
				pauseMining.store(true);
	    }
	}
    ).track_foreign(m_zmt)); // So the signal disconnects when the mining thread exits

	// Initialize context memory.
	void* context_alloc = malloc(CONTEXT_SIZE+4096);
	void* context = (void*) (((long long)context_alloc+((long long)4095)) & -((long long)4096));

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
			CBlockHeader actualHeader;
			std::string actualJobId;
			std::string actualTime;
			size_t actualNonce1size;
	    {
		std::lock_guard<std::mutex> lock{*m_zmt.get()};
		arith_uint256 baseNonce = UintToArith256(header.nNonce);
				arith_uint256 add(pos);
				nonce = baseNonce | (add << (8 * 31));
				nonceEnd = baseNonce | ((add + 1) << (8 * 31));
				//nonce = baseNonce + ((space/size)*pos << offset);
				//nonceEnd = baseNonce + ((space/size)*(pos+1) << offset);

				// save job id and time
				actualHeader = header;
				actualJobId = jobId;
				actualTime = nTime;
				actualNonce1size = offset / 4;
	    }

			// I = the block header minus nonce and solution.
			CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
			{
				//std::lock_guard<std::mutex> lock{ *m_zmt.get() };
				CEquihashInput I{ actualHeader };
				ss << I;
			}

			const char *tequihash_header = (char *)&ss[0];
			unsigned int tequihash_header_len = ss.size();

	    // Start working
	    while (true) {
				BOOST_LOG_CUSTOM(debug, pos) << "Running Equihash solver with nNonce = " << nonce.ToString();

				auto bNonce = ArithToUint256(nonce);
		std::function<bool(std::vector<unsigned char>)> validBlock =
					[&m_zmt, &actualHeader, &bNonce, &target, &miner, pos, &actualJobId, &actualTime, &actualNonce1size]
			(std::vector<unsigned char> soln) {
		    //std::lock_guard<std::mutex> lock{*m_zmt.get()};
		    // Write the solution to the hash and compute the result.
					BOOST_LOG_CUSTOM(debug, pos) << "Checking solution against target...";
					actualHeader.nNonce = bNonce;
					actualHeader.nSolution = soln;

					speed.AddSolution();

					uint256 headerhash = actualHeader.GetHash();
					if (UintToArith256(headerhash) > target) {
						BOOST_LOG_CUSTOM(debug, pos) << "Too large: " << headerhash.ToString();
			return false;
		    }

		    // Found a solution
					BOOST_LOG_CUSTOM(debug, pos) << "Found solution with header hash: " << headerhash.ToString();
					EquihashSolution solution{ bNonce, soln, actualTime, actualNonce1size };
		    miner->submitSolution(solution, actualJobId);

		    // We're a pooled miner, so try all solutions
		    return false;
		};

				//////////////////////////////////////////////////////////////////////////
				// Xenoncat solver.
				/////////////////////////////////////////////////////////////////////////
				// bnonce is 32 bytes, read last four bytes as nonce int and send it to
				// eh solver method.
					unsigned char *tequihash_header = (unsigned char *)&ss[0];
					unsigned int tequihash_header_len = ss.size();
					unsigned char inputheader[144];
					memcpy(inputheader, tequihash_header, tequihash_header_len);

					// Write 32 byte nonce to input header.
					uint256 arthNonce = ArithToUint256(nonce);
					memcpy(inputheader + tequihash_header_len, (unsigned  char*) arthNonce.begin(), arthNonce.size());


					(*EhPrepare)(context, (void *) inputheader);

				unsigned char* nonceBegin = bNonce.begin();
				uint32_t nonceToApi = *(uint32_t *)(nonceBegin+28);
				uint32_t numsolutions = (*EhSolver)(context, nonceToApi);
				if (!cancelSolver.load()) {
					for (uint32_t i=0; i<numsolutions; i++) {
						// valid block method expects vector of unsigned chars.
						unsigned char* solutionStart = (unsigned char*)(((unsigned char*)context)+1344*i);
						unsigned char* solutionEnd = solutionStart + 1344;
						std::vector<unsigned char> solution(solutionStart, solutionEnd);
						validBlock(solution);
					}
				}
						speed.AddHash(); // Metrics, add one hash execution.

				//////////////////////////////////////////////////////////////////////////
				// Xenoncat solver.
				/////////////////////////////////////////////////////////////////////////
		// Check for stop
				if (!miner->minerThreadActive[pos])
					throw boost::thread_interrupted();
		//boost::this_thread::interruption_point();

				// Update nonce
				nonce += inc;

		if (nonce == nonceEnd) {
		    break;
		}

		// Check for new work
		if (workReady.load()) {
					BOOST_LOG_CUSTOM(debug, pos) << "New work received, dropping current work";
		    break;
		}

				if (pauseMining.load())
				{
					BOOST_LOG_CUSTOM(debug, pos) << "Mining paused";
					break;
				}
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
    // Free the memory allocated previously for xenoncat context.
    free(context_alloc);
}
#endif

void static TrompZcashMinerThread(ZcashMiner* miner, int size, int pos)
{
	BOOST_LOG_CUSTOM(info, pos) << "Starting thread #" << pos;

	unsigned int n = PARAMETER_N;
	unsigned int k = PARAMETER_K;

    std::shared_ptr<std::mutex> m_zmt(new std::mutex);
    CBlockHeader header;
    arith_uint256 space;
    size_t offset;
    arith_uint256 inc;
    arith_uint256 target;
	std::string jobId;
	std::string nTime;
    std::atomic_bool workReady {false};
    std::atomic_bool cancelSolver {false};
	std::atomic_bool pauseMining {false};

    miner->NewJob.connect(NewJob_t::slot_type(
		[&m_zmt, &header, &space, &offset, &inc, &target, &workReady, &cancelSolver, pos, &pauseMining, &jobId, &nTime]
        (const ZcashJob* job) mutable {
            std::lock_guard<std::mutex> lock{*m_zmt.get()};
            if (job) {
				BOOST_LOG_CUSTOM(debug, pos) << "Loading new job #" << job->jobId();
				jobId = job->jobId();
				nTime = job->time;
                header = job->header;
                space = job->nonce2Space;
                offset = job->nonce1Size * 4; // Hex length to bit length
                inc = job->nonce2Inc;
                target = job->serverTarget;
				pauseMining.store(false);
                workReady.store(true);
                /*if (job->clean) {
                    cancelSolver.store(true);
                }*/
            } else {
                workReady.store(false);
                cancelSolver.store(true);
				pauseMining.store(true);
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
			CBlockHeader actualHeader;
			std::string actualJobId;
			std::string actualTime;
			size_t actualNonce1size;
            {
                std::lock_guard<std::mutex> lock{*m_zmt.get()};
                arith_uint256 baseNonce = UintToArith256(header.nNonce);
				arith_uint256 add(pos);
				nonce = baseNonce | (add << (8 * 31));
				nonceEnd = baseNonce | ((add + 1) << (8 * 31));
				//nonce = baseNonce + ((space/size)*pos << offset);
				//nonceEnd = baseNonce + ((space/size)*(pos+1) << offset);

				// save job id and time
				actualHeader = header;
				actualJobId = jobId;
				actualTime = nTime;
				actualNonce1size = offset / 4;
            }

			// I = the block header minus nonce and solution.
			CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
			{
				//std::lock_guard<std::mutex> lock{ *m_zmt.get() };
				CEquihashInput I{ actualHeader };
				ss << I;
			}

			const char *tequihash_header = (char *)&ss[0];
			unsigned int tequihash_header_len = ss.size();

            // Start working
            while (true) {
				BOOST_LOG_CUSTOM(debug, pos) << "Running Equihash solver with nNonce = " << nonce.ToString();

				auto bNonce = ArithToUint256(nonce);
                std::function<bool(std::vector<unsigned char>)> validBlock =
					[&m_zmt, &actualHeader, &bNonce, &target, &miner, pos, &actualJobId, &actualTime, &actualNonce1size]
                        (std::vector<unsigned char> soln) {
                    //std::lock_guard<std::mutex> lock{*m_zmt.get()};
                    // Write the solution to the hash and compute the result.
					BOOST_LOG_CUSTOM(debug, pos) << "Checking solution against target...";
					actualHeader.nNonce = bNonce;
					actualHeader.nSolution = soln;

					speed.AddSolution();

					uint256 headerhash = actualHeader.GetHash();
					if (UintToArith256(headerhash) > target) {
						BOOST_LOG_CUSTOM(debug, pos) << "Too large: " << headerhash.ToString();
                        return false;
                    }

                    // Found a solution
					BOOST_LOG_CUSTOM(debug, pos) << "Found solution with header hash: " << headerhash.ToString();
					EquihashSolution solution{ bNonce, soln, actualTime, actualNonce1size };
                    miner->submitSolution(solution, actualJobId);

                    // We're a pooled miner, so try all solutions
                    return false;
                };

				//////////////////////////////////////////////////////////////////////////
				// TROMP EQ SOLVER START
				// I = the block header minus nonce and solution.
				// Nonce
				// Create solver and initialize it with header and nonce.
				equi eq(1);
				eq.setnonce(tequihash_header, tequihash_header_len, (const char*)bNonce.begin(), bNonce.size());
				eq.digit0(0);
				eq.xfull = eq.bfull = eq.hfull = 0;
				eq.showbsizes(0);
				u32 r = 1;
				for ( ; r < WK; r++) {
					if (cancelSolver.load()) break;
					r & 1 ? eq.digitodd(r, 0) : eq.digiteven(r, 0);
					eq.xfull = eq.bfull = eq.hfull = 0;
					eq.showbsizes(r);
				}
				if (r == WK && !cancelSolver.load())
				{
					eq.digitK(0);

					// Convert solution indices to charactar array(decompress) and pass it to validBlock method.
					u32 nsols = 0;
					unsigned s = 0;
					for (; s < eq.nsols; s++)
					{
						if (cancelSolver.load()) break;
						nsols++;
						std::vector<eh_index> index_vector(PROOFSIZE);
						for (u32 i = 0; i < PROOFSIZE; i++) {
							index_vector[i] = eq.sols[s][i];
						}
						std::vector<unsigned char> sol_char = GetMinimalFromIndices(index_vector, DIGITBITS);

						if (validBlock(sol_char))
						{
							// If we find a POW solution, do not try other solutions
							// because they become invalid as we created a new block in blockchain.
							//break;
						}
					}
					if (s == eq.nsols)
						speed.AddHash();
				}
				//////////////////////////////////////////////////////////////////////
				// TROMP EQ SOLVER END
				//////////////////////////////////////////////////////////////////////
				
                // Check for stop
				if (!miner->minerThreadActive[pos])
					throw boost::thread_interrupted();
                //boost::this_thread::interruption_point();

				// Update nonce
				nonce += inc;

                if (nonce == nonceEnd) {
                    break;
                }

                // Check for new work
                if (workReady.load()) {
					BOOST_LOG_CUSTOM(debug, pos) << "New work received, dropping current work";
                    break;
                }

				if (pauseMining.load())
				{
					BOOST_LOG_CUSTOM(debug, pos) << "Mining paused";
					break;
				}
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

// Windows have __cpuidex
#ifdef _WIN32
#define cpuid(info, x)    __cpuidex(info, x, 0)
#endif

#ifdef __clang__
void cpuid(int32_t out[4], int32_t x){
	__cpuid_count(x, 0, out[0], out[1], out[2], out[3]);
}
#endif

int detect_avx (void) {
	// return engine forcefully set via commandline
	if (equiengine>-1) {
		return equiengine;
	}
	// if not set on commandline, auto-detect it!
#if (!defined(__NONINTEL__)) && (defined(__GNUC__) || defined(__MINGW32__))
#ifndef __clang__
	if (__builtin_cpu_supports("avx2")) {
		return 2;
	}
	if (__builtin_cpu_supports("avx")) {
		return 1;
	}
#endif // __clang__
#endif // __GNUC__
#if defined(__clang__) // clang does not have __builtin_cpu_supports
	int info[4];
	cpuid(info, 0);
	int nIds = info[0];

	cpuid(info, 0x80000000);
	uint32_t nExIds = info[0];
	// AVX2
	if (nIds >= 0x00000007){
		cpuid(info, 0x00000007);
		if ((info[1] & ((int)1 << 5)) != 0) {
			return 2;
		}
	}
	// AVX1
	if (nIds >= 0x00000001){
		cpuid(info, 0x00000001);
		if ((info[2] & ((int)1 << 28)) != 0) {
			return 1;
		}
	}
#endif
	// Fallback to no-AVX
	return 0;
}

void static ZcashMinerThread(ZcashMiner* miner, int size, int pos)
{

	#ifdef XENONCAT
		if (detect_avx()==2) {
			BOOST_LOG_CUSTOM(info, pos) << "Using Xenoncat's AVX2 solver. ";
			EhPrepare=&EhPrepareAVX2;
			EhSolver=&EhSolverAVX2;
			XenoncatZcashMinerThread(miner, size, pos);
		}
		else if (detect_avx()==1) {
			BOOST_LOG_CUSTOM(info, pos) << "Using Xenoncat's AVX solver. ";
			EhPrepare=&EhPrepareAVX1;
			EhSolver=&EhSolverAVX1;
			XenoncatZcashMinerThread(miner, size, pos);
		} else {
	#endif
		BOOST_LOG_CUSTOM(info, pos) << "Using Tromp's solver.";
		TrompZcashMinerThread(miner, size, pos);
	#ifdef XENONCAT
	}
	#endif
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
		nThreads = std::thread::hardware_concurrency() * 3 / 4; // take 75% of all threads by default
		if (nThreads < 1) nThreads = 1;
    }
}

std::string ZcashMiner::userAgent()
{
	return STANDALONE_MINER_NAME "/" STANDALONE_MINER_VERSION;
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
		for (int i = 0; i < nThreads; i++) {
			BOOST_LOG_CUSTOM(warning, i) << "Waiting for miners join";
			while (minerThreads[i].joinable()) { boost::this_thread::sleep(boost::posix_time::milliseconds(100)); } ;
		}

		delete[] minerThreads;
		delete[] minerThreadActive;
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

	//BOOST_LOG_TRIVIAL(info) << "miner | Full nonce " << nonce1.ToString();

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
        const std::function<bool(const EquihashSolution&, const std::string&)> callback)
{
    solutionFoundCallback = callback;
}

void ZcashMiner::submitSolution(const EquihashSolution& solution, const std::string& jobid)
{
    solutionFoundCallback(solution, jobid);
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


std::mutex benchmark_work;
std::vector<uint256*> benchmark_nonces;
std::atomic_int benchmark_solutions;

bool benchmark_solve_equihash()
{
	CBlock pblock;
	CEquihashInput I{ pblock };
	CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
	ss << I;

	unsigned int n = PARAMETER_N;
	unsigned int k = PARAMETER_K;

	const char *tequihash_header = (char *)&ss[0];
	unsigned int tequihash_header_len = ss.size();

	benchmark_work.lock();
	if (benchmark_nonces.empty())
	{
		benchmark_work.unlock();
		return false;
	}
	uint256* nonce = benchmark_nonces.front();
	benchmark_nonces.erase(benchmark_nonces.begin());
	benchmark_work.unlock();

	BOOST_LOG_TRIVIAL(debug) << "Testing, nonce = " << nonce->ToString();

	equi eq(1);
	eq.setnonce(tequihash_header, tequihash_header_len, (const char*)nonce->begin(), nonce->size());
	eq.digit0(0);
	eq.xfull = eq.bfull = eq.hfull = 0;
	eq.showbsizes(0);
	u32 r = 1;
	for ( ; r < WK; r++) {
		r & 1 ? eq.digitodd(r, 0) : eq.digiteven(r, 0);
		eq.xfull = eq.bfull = eq.hfull = 0;
		eq.showbsizes(r);
	}

	eq.digitK(0);

	u32 nsols = 0;
	unsigned s = 0;
	for (; s < eq.nsols; s++)
	{
		nsols++;
		std::vector<eh_index> index_vector(PROOFSIZE);
		for (u32 i = 0; i < PROOFSIZE; i++) {
			index_vector[i] = eq.sols[s][i];
		}
		std::vector<unsigned char> sol_char = GetMinimalFromIndices(index_vector, DIGITBITS);
		
		CBlockHeader hdr = pblock.GetBlockHeader();
		hdr.nNonce = *nonce;
		hdr.nSolution = sol_char;

		BOOST_LOG_TRIVIAL(debug) << "Solution found, header = " << hdr.GetHash().ToString();

		++benchmark_solutions;
	}

	delete nonce;

	return true;
}


int benchmark_thread(int tid)
{
	BOOST_LOG_TRIVIAL(debug) << "Thread #" << tid << " started";

	while (benchmark_solve_equihash()) {}

	BOOST_LOG_TRIVIAL(debug) << "Thread #" << tid << " ended";

	return 0;
}


void do_benchmark(int nThreads, int hashes)
{
	// generate array of various nonces
	std::srand(std::time(0));
	for (int i = 0; i < hashes; ++i)
	{
		benchmark_nonces.push_back(new uint256());
		for (unsigned int i = 0; i < 32; ++i)
			benchmark_nonces.back()->begin()[i] = std::rand() % 256;
	}
	benchmark_solutions = 0;

	size_t total_hashes = benchmark_nonces.size();

	std::cout << "Benchmark starting... this may take several minutes, please wait..." << std::endl;

	if (nThreads < 1)
	{
		nThreads = std::thread::hardware_concurrency() * 3 / 4; // take 75% of all threads by default
		if (nThreads < 1) nThreads = 1;
	}
	std::thread* bthreads = new std::thread[nThreads];

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < nThreads; ++i)
		bthreads[i] = std::thread(boost::bind(&benchmark_thread, i));

	for (int i = 0; i < nThreads; ++i)
		bthreads[i].join();

	auto end = std::chrono::high_resolution_clock::now();

	uint64_t msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	size_t hashes_done = total_hashes - benchmark_nonces.size();

	std::cout << "Benchmark done!" << std::endl;
	std::cout << "Total time : " << msec << " ms" << std::endl;
	std::cout << "Total hashes: " << hashes_done << std::endl;
	std::cout << "Total solutions found: " << benchmark_solutions << std::endl;
	std::cout << "Speed: " << ((double)hashes_done * 1000 / (double)msec) << " H/s" << std::endl;
	std::cout << "Speed: " << ((double)benchmark_solutions * 1000 / (double)msec) << " S/s" << std::endl;
}
