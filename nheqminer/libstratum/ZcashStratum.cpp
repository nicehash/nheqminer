// Copyright (c) 2016 Jack Grigg <jack@z.cash>
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "version.h"
#include "ZcashStratum.h"

#include "utilstrencodings.h"
//#include "trompequihash/equi_miner.h"
#include "streams.h"

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

#include <boost/static_assert.hpp>


typedef uint32_t eh_index;


#define BOOST_LOG_CUSTOM(sev, pos) BOOST_LOG_TRIVIAL(sev) << "miner#" << pos << " | "


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

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver, typename Solver>
void static ZcashMinerThread(ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>* miner, int size, int pos, Solver& extra)
{
	BOOST_LOG_CUSTOM(info, pos) << "Starting thread #" << pos << " (" << extra.getname() << ") " << extra.getdevinfo();

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

		Solver::start(extra);

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
			arith_uint256 actualTarget;
			size_t actualNonce1size;
            {
                std::lock_guard<std::mutex> lock{*m_zmt.get()};
                arith_uint256 baseNonce = UintToArith256(header.nNonce);
				arith_uint256 add(pos);
				nonce = baseNonce | (add << (8 * 19));
				nonceEnd = baseNonce | ((add + 1) << (8 * 19));
				//nonce = baseNonce + ((space/size)*pos << offset);
				//nonceEnd = baseNonce + ((space/size)*(pos+1) << offset);

				// save job id and time
				actualHeader = header;
				actualJobId = jobId;
				actualTime = nTime;
				actualNonce1size = offset / 4;
				actualTarget = target;
            }

			// I = the block header minus nonce and solution.
			CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
			{
				//std::lock_guard<std::mutex> lock{ *m_zmt.get() };
				CEquihashInput I{ actualHeader };
				ss << I;
			}

			char *tequihash_header = (char *)&ss[0];
			unsigned int tequihash_header_len = ss.size();

            // Start working
            while (true) {
				BOOST_LOG_CUSTOM(debug, pos) << "Running Equihash solver with nNonce = " << nonce.ToString();

				auto bNonce = ArithToUint256(nonce);

				std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionFound =
					[&actualHeader, &bNonce, &actualTarget, &miner, pos, &actualJobId, &actualTime, &actualNonce1size]
				(const std::vector<uint32_t>& index_vector, size_t cbitlen, const unsigned char* compressed_sol) 
				{
					actualHeader.nNonce = bNonce;
					if (compressed_sol)
					{
						actualHeader.nSolution = std::vector<unsigned char>(1344);
						for (size_t i = 0; i < cbitlen; ++i)
							actualHeader.nSolution[i] = compressed_sol[i];
					}
					else
						actualHeader.nSolution = GetMinimalFromIndices(index_vector, cbitlen);

					speed.AddSolution();

					BOOST_LOG_CUSTOM(debug, pos) << "Checking solution against target...";

					uint256 headerhash = actualHeader.GetHash();
					if (UintToArith256(headerhash) > actualTarget) {
						BOOST_LOG_CUSTOM(debug, pos) << "Too large: " << headerhash.ToString();
						return;
					}

					// Found a solution
					BOOST_LOG_CUSTOM(debug, pos) << "Found solution with header hash: " << headerhash.ToString();
					EquihashSolution solution{ bNonce, actualHeader.nSolution, actualTime, actualNonce1size };
					miner->submitSolution(solution, actualJobId);
				};

				std::function<bool()> cancelFun = [&cancelSolver]() {
					return cancelSolver.load();
				};

				std::function<void(void)> hashDone = []() {
					speed.AddHash();
				};

				Solver::solve(tequihash_header,
					tequihash_header_len,
					(const char*)bNonce.begin(),
					bNonce.size(),
					cancelFun,
					solutionFound,
					hashDone,
					extra);
				
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
        //throw;
    }
    catch (const std::runtime_error &e)
    {
		BOOST_LOG_CUSTOM(error, pos) << e.what();
    }

	try
	{
		Solver::stop(extra);
	}
	catch (const std::runtime_error &e)
	{
		BOOST_LOG_CUSTOM(error, pos) << e.what();
	}

	BOOST_LOG_CUSTOM(info, pos) << "Thread #" << pos << " ended (" << extra.getname() << ")";
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

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::ZcashMiner(int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t, 
	int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t)
    : minerThreads{nullptr}
{
	m_isActive = false;
    nThreads = 0;

    for (int i = 0; i < cuda_count; ++i)
    {
        CUDASolver* context = new CUDASolver(0, cuda_en[i]);
        if (cuda_b[i] > 0)
            context->blocks = cuda_b[i];
        if (cuda_t[i] > 0)
            context->threadsperblock = cuda_t[i];

        cuda_contexts.push_back(context);
    }
    nThreads += cuda_contexts.size();


    for (int i = 0; i < opencl_count; ++i)
    {
		if (opencl_t[i] < 1) opencl_t[i] = 1;

		// add multiple threads if wanted
		for (int k = 0; k < opencl_t[i]; ++k)
		{
			OPENCLSolver* context = new OPENCLSolver(opencl_platf, opencl_en[i]);
			// todo: save local&global work size
			opencl_contexts.push_back(context);
		}
    }
    nThreads += opencl_contexts.size();



    if (cpu_threads < 0) {
        cpu_threads = std::thread::hardware_concurrency();
        if (cpu_threads < 1) cpu_threads = 1;
        else if (cuda_contexts.size() + opencl_contexts.size() > 0) --cpu_threads; // decrease number of threads if there are GPU workers
    }


    for (int i = 0; i < cpu_threads; ++i)
    {
        CPUSolver* context = new CPUSolver();
        context->use_opt = use_avx2;
        cpu_contexts.push_back(context);
    }
    nThreads += cpu_contexts.size();


//	nThreads = cpu_contexts.size() + cuda_contexts.size() + opencl_contexts.size();
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::~ZcashMiner()
{
    stop();
    for (auto it = cpu_contexts.begin(); it != cpu_contexts.end(); ++it)
        delete (*it);
    for (auto it = cuda_contexts.begin(); it != cuda_contexts.end(); ++it)
        delete (*it);
	cpu_contexts.clear();
	cuda_contexts.clear();
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
std::string ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::userAgent()
{
	return "equihashminer/" STANDALONE_MINER_VERSION;
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::start()
{
    if (minerThreads) {
        stop();
    }

	m_isActive = true;

	minerThreads = new std::thread[nThreads];
	minerThreadActive = new bool[nThreads];


    // start cpu threads
    int i = 0;
    for ( ; i < cpu_contexts.size(); ++i)
    {
        minerThreadActive[i] = true;
        minerThreads[i] = std::thread(boost::bind(&ZcashMinerThread<CPUSolver, CUDASolver, OPENCLSolver, CPUSolver>,
            this, nThreads, i, *cpu_contexts.at(i)));
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



    // start CUDA threads
    for (; i < (cpu_contexts.size() + cuda_contexts.size()); ++i)
    {
        minerThreadActive[i] = true;
        minerThreads[i] = std::thread(boost::bind(&ZcashMinerThread<CPUSolver, CUDASolver, OPENCLSolver, CUDASolver>,
            this, nThreads, i, *cuda_contexts.at(i - cpu_contexts.size())));
    }



    // start OPENCL threads
    for (; i < (cpu_contexts.size() + cuda_contexts.size() + opencl_contexts.size()); ++i)
    {
        minerThreadActive[i] = true;
        minerThreads[i] = std::thread(boost::bind(&ZcashMinerThread<CPUSolver, CUDASolver, OPENCLSolver, OPENCLSolver>,
            this, nThreads, i, *opencl_contexts.at(i - cpu_contexts.size() - cuda_contexts.size())));
    }


    /*minerThreads = new boost::thread_group();
    for (int i = 0; i < nThreads; i++) {
        minerThreads->create_thread(boost::bind(&ZcashMinerThread, this, nThreads, i));
    }*/

	speed.Reset();
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::stop()
{
	m_isActive = false;
	if (minerThreads)
	{
		for (int i = 0; i < nThreads; i++)
			minerThreadActive[i] = false;
		for (int i = 0; i < nThreads; i++)
			minerThreads[i].join();
		delete minerThreads;
		minerThreads = nullptr;
		delete minerThreadActive;
	}
    /*if (minerThreads) {
        minerThreads->interrupt_all();
        delete minerThreads;
        minerThreads = nullptr;
    }*/
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::setServerNonce(const std::string& n1str)
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

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
ZcashJob* ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::parseJob(const Array& params)
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

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::setJob(ZcashJob* job)
{
    NewJob(job);
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::onSolutionFound(
        const std::function<bool(const EquihashSolution&, const std::string&)> callback)
{
    solutionFoundCallback = callback;
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::submitSolution(const EquihashSolution& solution, const std::string& jobid)
{
    solutionFoundCallback(solution, jobid);
	speed.AddShare();
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::acceptedSolution(bool stale)
{
	speed.AddShareOK();
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::rejectedSolution(bool stale)
{
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::failedSolution()
{
}

// XMP
template class ZcashMiner<cpu_xenoncat, cuda_tromp, ocl_xmp>;
template class ZcashMiner<cpu_tromp, cuda_tromp, ocl_xmp>;
template class ZcashMiner<cpu_xenoncat, cuda_tromp_75, ocl_xmp>;
template class ZcashMiner<cpu_tromp, cuda_tromp_75, ocl_xmp>;
template class ZcashMiner<cpu_xenoncat, cuda_sa_solver, ocl_xmp>;
template class ZcashMiner<cpu_tromp, cuda_sa_solver, ocl_xmp>;

// Silentarmy
template class ZcashMiner<cpu_xenoncat, cuda_tromp, ocl_silentarmy>;
template class ZcashMiner<cpu_tromp, cuda_tromp, ocl_silentarmy>;
template class ZcashMiner<cpu_xenoncat, cuda_tromp_75, ocl_silentarmy>;
template class ZcashMiner<cpu_tromp, cuda_tromp_75, ocl_silentarmy>;
template class ZcashMiner<cpu_xenoncat, cuda_sa_solver, ocl_silentarmy>;
template class ZcashMiner<cpu_tromp, cuda_sa_solver, ocl_silentarmy>;

std::mutex benchmark_work;
std::vector<uint256*> benchmark_nonces;
std::atomic_int benchmark_solutions;

template <typename Solver>
bool benchmark_solve_equihash(const CBlock& pblock, const char *tequihash_header, unsigned int tequihash_header_len, Solver& extra)
{
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

	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionFound =
		[&pblock, &nonce]
	(const std::vector<uint32_t>& index_vector, size_t cbitlen, const unsigned char* compressed_sol)
	{
		CBlockHeader hdr = pblock.GetBlockHeader();
		hdr.nNonce = *nonce;

		if (compressed_sol)
		{
			hdr.nSolution = std::vector<unsigned char>(1344);
			for (size_t i = 0; i < cbitlen; ++i)
				hdr.nSolution[i] = compressed_sol[i];
		}
		else
			hdr.nSolution = GetMinimalFromIndices(index_vector, cbitlen);

		BOOST_LOG_TRIVIAL(debug) << "Solution found, header = " << hdr.GetHash().ToString();

		++benchmark_solutions;
	};

	Solver::solve(tequihash_header,
		tequihash_header_len,
		(const char*)nonce->begin(),
		nonce->size(),
		[]() { return false; },
		solutionFound,
		[]() {},
		extra);

	delete nonce;

	return true;
}

template <typename Solver>
int benchmark_thread(int tid, Solver& extra)
{
	BOOST_LOG_TRIVIAL(debug) << "Thread #" << tid << " started (" << extra.getname() << ")";

	try
	{
		CBlock pblock;
		CEquihashInput I{ pblock };
		CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
		ss << I;

		const char *tequihash_header = (char *)&ss[0];
		unsigned int tequihash_header_len = ss.size();

		while (benchmark_solve_equihash<Solver>(pblock, tequihash_header, tequihash_header_len, extra)) {}
	}
	catch (const std::runtime_error &e)
	{
		BOOST_LOG_TRIVIAL(error) << e.what();
		exit(0);
		return 0;
	}

	BOOST_LOG_TRIVIAL(debug) << "Thread #" << tid << " ended (" << extra.getname() << ")";

	return 0;
}

template <typename CPUSolver, typename CUDASolver, typename OPENCLSolver>
void ZcashMiner<CPUSolver, CUDASolver, OPENCLSolver>::doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
	int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t)
{
	// generate array of various nonces
	std::srand(std::time(0));
	benchmark_nonces.push_back(new uint256());
	benchmark_nonces.back()->begin()[31] = 1;
	for (int i = 0; i < (hashes - 1); ++i)
	{
		benchmark_nonces.push_back(new uint256());
		for (unsigned int i = 0; i < 32; ++i)
			benchmark_nonces.back()->begin()[i] = std::rand() % 256;
	}
	benchmark_solutions = 0;

	size_t total_hashes = benchmark_nonces.size();

	std::vector<CPUSolver*> cpu_contexts;
	std::vector<CUDASolver*> cuda_contexts;
	std::vector<OPENCLSolver*> opencl_contexts;

	for (int i = 0; i < cuda_count; ++i)
	{
		CUDASolver* context = new CUDASolver(0, cuda_en[i]);
		if (cuda_b[i] > 0)
			context->blocks = cuda_b[i];
		if (cuda_t[i] > 0)
			context->threadsperblock = cuda_t[i];

		BOOST_LOG_TRIVIAL(info) << "Benchmarking CUDA worker (" << context->getname() << ") " << context->getdevinfo();

		CUDASolver::start(*context); // init CUDA before to get more accurate benchmark

		cuda_contexts.push_back(context);
	}

	for (int i = 0; i < opencl_count; ++i)
	{
		if (opencl_t[i] < 1) opencl_t[i] = 1;

		for (int k = 0; k < opencl_t[i]; ++k)
		{
			OPENCLSolver* context = new OPENCLSolver(opencl_platf, opencl_en[i]);

			// todo: save local&global work size

			BOOST_LOG_TRIVIAL(info) << "Benchmarking OPENCL worker (" << context->getname() << ") " << context->getdevinfo();

			OPENCLSolver::start(*context); // init OPENCL before to get more accurate benchmark

			opencl_contexts.push_back(context);
		}
	}

	if (cpu_threads < 0)
	{
		cpu_threads = std::thread::hardware_concurrency();
		if (cpu_threads < 1) cpu_threads = 1;
		else if (cuda_contexts.size() + opencl_contexts.size() > 0) --cpu_threads; // decrease number of threads if there are GPU workers
	}

	for (int i = 0; i < cpu_threads; ++i)
	{
		CPUSolver* context = new CPUSolver();
		context->use_opt = use_avx2;
		BOOST_LOG_TRIVIAL(info) << "Benchmarking CPU worker (" << context->getname() << ") " << context->getdevinfo();
		CPUSolver::start(*context);
		cpu_contexts.push_back(context);
	}

	int nThreads = cpu_contexts.size() + cuda_contexts.size() + opencl_contexts.size();

	std::thread* bthreads = new std::thread[nThreads];

	BOOST_LOG_TRIVIAL(info) << "Benchmark starting... this may take several minutes, please wait...";

	auto start = std::chrono::high_resolution_clock::now();

	int i = 0;
	for ( ; i < cpu_contexts.size(); ++i)
		bthreads[i] = std::thread(boost::bind(&benchmark_thread<CPUSolver>, i, *cpu_contexts.at(i)));

	for (; i < (cuda_contexts.size() + cpu_contexts.size()); ++i)
		bthreads[i] = std::thread(boost::bind(&benchmark_thread<CUDASolver>, i, *cuda_contexts.at(i - cpu_contexts.size())));

	for (; i < (opencl_contexts.size() + cuda_contexts.size() + cpu_contexts.size()); ++i)
		bthreads[i] = std::thread(boost::bind(&benchmark_thread<OPENCLSolver>, i, *opencl_contexts.at(i - cpu_contexts.size() - cuda_contexts.size())));

	for (int i = 0; i < nThreads; ++i)
		bthreads[i].join();

	auto end = std::chrono::high_resolution_clock::now();

	uint64_t msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	size_t hashes_done = total_hashes - benchmark_nonces.size();

    for (auto it = cpu_contexts.begin(); it != cpu_contexts.end(); ++it)
	{
		CPUSolver::stop(**it);
		delete (*it);
	}
    for (auto it = cuda_contexts.begin(); it != cuda_contexts.end(); ++it)
	{
		CUDASolver::stop(**it);
		delete (*it);
	}
    for (auto it = opencl_contexts.begin(); it != opencl_contexts.end(); ++it)
	{
		OPENCLSolver::stop(**it);
		delete (*it);
	}
	cpu_contexts.clear();
	cuda_contexts.clear();
	opencl_contexts.clear();

	BOOST_LOG_TRIVIAL(info) << "Benchmark done!";
	BOOST_LOG_TRIVIAL(info) << "Total time : " << msec << " ms";
	BOOST_LOG_TRIVIAL(info) << "Total iterations: " << hashes_done;
	BOOST_LOG_TRIVIAL(info) << "Total solutions found: " << benchmark_solutions;
	BOOST_LOG_TRIVIAL(info) << "Speed: " << ((double)hashes_done * 1000 / (double)msec) << " I/s";
	BOOST_LOG_TRIVIAL(info) << "Speed: " << ((double)benchmark_solutions * 1000 / (double)msec) << " Sols/s";
}


//void ZMinerAVX_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
//                           int opencl_count, int opencl_platf, int* opencl_en) {
//    ZMinerAVX::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en);
//}
//
//void ZMinerSSE2_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
//                            int opencl_count, int opencl_platf, int* opencl_en) {
//    ZMinerSSE2::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en);
//}



// ocl_xmp
// gcc static undefined reference workaround
void ZMinerAVXCUDA80_XMP_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerAVXCUDA80_XMP::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerSSE2CUDA80_XMP_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerSSE2CUDA80_XMP::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerAVXCUDA75_XMP_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerAVXCUDA75_XMP::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerSSE2CUDA75_XMP_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerSSE2CUDA75_XMP::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerAVXCUDASA80_XMP_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
	int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
	ZMinerAVXCUDASA80_XMP::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerSSE2CUDASA80_XMP_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
	int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
	ZMinerSSE2CUDASA80_XMP::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}

// ocl_silentarmy
void ZMinerAVXCUDA80_SA_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerAVXCUDA80_SA::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerSSE2CUDA80_SA_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerSSE2CUDA80_SA::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerAVXCUDA75_SA_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerAVXCUDA75_SA::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerSSE2CUDA75_SA_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
    int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
    ZMinerSSE2CUDA75_SA::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerAVXCUDASA80_SA_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
	int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
	ZMinerAVXCUDASA80_SA::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}
void ZMinerSSE2CUDASA80_SA_doBenchmark(int hashes, int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
	int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t) {
	ZMinerSSE2CUDASA80_SA::doBenchmark(hashes, cpu_threads, cuda_count, cuda_en, cuda_b, cuda_t, opencl_count, opencl_platf, opencl_en, opencl_t);
}

