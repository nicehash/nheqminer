#include <iostream>

#include "version.h"
#include "arith_uint256.h"
#include "primitives/block.h"
#include "streams.h"

#include "libstratum/StratumClient.h"

#include <thread>
#include <chrono>
#include <atomic>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>

#include "speed.hpp"
#include "api.hpp"


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
	
	blake2b_state eh_state;

	EhInitialiseState(n, k, eh_state);
	blake2b_update(&eh_state, (unsigned char*)&ss[0], ss.size());

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

	blake2b_update(&eh_state,
		nonce->begin(),
		nonce->size());
	
	std::set<std::vector<unsigned int>> solns;
	EhOptimisedSolveUncancellable(n, k, eh_state, [nonce, &pblock](std::vector<unsigned char> soln) 
	{
		CBlockHeader hdr = pblock.GetBlockHeader();
		hdr.nNonce = *nonce;
		hdr.nSolution = soln;

		BOOST_LOG_TRIVIAL(debug) << "Solution found, header = " << hdr.GetHash().ToString();

		++benchmark_solutions;

		return false;
	});

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

	if (nThreads < 1) nThreads = std::thread::hardware_concurrency();
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

static ZcashStratumClient* scSig;
extern "C" void stratum_sigint_handler(int signum) 
{ 
	if (scSig) scSig->disconnect(); 
}

void print_help()
{
	std::cout << "Parameters: " << std::endl;
	std::cout << "\t-h\t\tPrint this help and quit" << std::endl;
	std::cout << "\t-l [location]\tLocation (eu, usa, hk, jp)" << std::endl;
	std::cout << "\t-u [username]\tUsername (bitcoinaddress)" << std::endl;
	std::cout << "\t-p [password]\tPassword (default: x)" << std::endl;
	std::cout << "\t-t [num_thrds]\tNumber of threads (default: number of sys cores)" << std::endl;
	std::cout << "\t-d [level]\tDebug print level (0 = print all, 5 = fatal only, default: 2)" << std::endl;
	std::cout << "\t-b [hashes]\tRun in benchmark mode (default: 20 hashes)" << std::endl;
	std::cout << "\t-a [port]\tLocal API port (default: 0 = do not bind)" << std::endl;
	std::cout << std::endl;
}


void init_logging(boost::log::core_ptr cptr, int level);


int main(int argc, char* argv[])
{
	std::cout << "Equihash CPU Miner for NiceHash" << std::endl;
	std::cout << "Special thanks to Zcash developers for providing most of the code" << std::endl;
	std::cout << std::endl;

	std::string location = "eu";
	std::string user = "1DXnVXrTmcEd77Z6E4zGxkn7fGeHXSGDt1";
	std::string password = "x";
	int num_threads = -1;
	bool benchmark = false;
	int log_level = 2;
	int num_hashes = 20;
	int api_port = 0;

	for (int i = 1; i < argc; ++i)
	{
		if (argv[i][0] != '-') continue;

		switch (argv[i][1])
		{
		case 'l':
			location = argv[++i];
			break;
		case 'u':
			user = argv[++i];
			break;
		case 'p':
			password = argv[++i];
			break;
		case 't':
			num_threads = atoi(argv[++i]);
			break;
		case 'h':
			print_help();
			return 0;
		case 'b':
			benchmark = true;
			num_hashes = atoi(argv[++i]);
			break;
		case 'd':
			log_level = atoi(argv[++i]);
			break;
		case 'a':
			api_port = atoi(argv[++i]);
			break;
		}
	}

	init_logging(boost::log::core::get(), log_level);

	if (!benchmark)
	{
		std::string host = "equihash." + location + ".nicehash.com";
		std::string port = "3357";

		std::shared_ptr<boost::asio::io_service> io_service(new boost::asio::io_service);

		API* api = nullptr;
		if (api_port > 0)
		{
			api = new API(io_service);
			if (!api->start(api_port))
			{
				delete api;
				api = nullptr;
			}
		}
		
		ZcashMiner miner(num_threads);
		ZcashStratumClient sc{
			io_service, &miner, host, port, user, password, 0, 0
		};

		miner.onSolutionFound([&](const EquihashSolution& solution) {
			return sc.submit(&solution);
		});

		scSig = &sc;
		signal(SIGINT, stratum_sigint_handler);

		int c = 0;
		while (sc.isRunning()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			if (++c % 1000 == 0)
			{
				double allshares = speed.GetShareSpeed() * 60;
				double accepted = speed.GetShareOKSpeed() * 60;
				BOOST_LOG_TRIVIAL(info) << CL_YLW "Speed [" << INTERVAL_SECONDS << " sec]: " << 
					speed.GetHashSpeed() << " H/s, " <<
					speed.GetHashInterruptedSpeed() << " IH/s, " << 
					speed.GetSolutionSpeed() << " S/s, " << 
					accepted << " AS/min, " << 
					(allshares - accepted) << " RS/min" CL_N;
			}
			if (api) while (api->poll()) { }
		}

		if (api) delete api;
	}
	else
	{
		do_benchmark(num_threads, num_hashes);
	}

	boost::log::core::get()->remove_all_sinks();

	return 0;
}