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


// TODO:
// bug: share above target right after new job arrival
// file logging
// mingw compilation for windows (faster?)
// fix SSE2 VS2013 compiler error (error C2105: '--' needs l-value)


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
	std::cout << "\t-b [hashes]\tRun in benchmark mode (default: 100 hashes)" << std::endl;
	std::cout << "\t-a [port]\tLocal API port (default: 0 = do not bind)" << std::endl;
	std::cout << std::endl;
}


void init_logging(boost::log::core_ptr cptr, int level);


int main(int argc, char* argv[])
{
	std::cout << "Equihash CPU Miner for NiceHash v" STANDALONE_MINER_VERSION << std::endl;
	std::cout << "Thanks to Zcash developers for providing most of the code" << std::endl;
	std::cout << "Special thanks to tromp for providing optimized CPU equihash solver" << std::endl;
	std::cout << std::endl;

	std::string location = "eu";
	std::string user = "1DXnVXrTmcEd77Z6E4zGxkn7fGeHXSGDt1";
	std::string password = "x";
	int num_threads = -1;
	bool benchmark = false;
	int log_level = 2;
	int num_hashes = 100;
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
			if (argv[i + 1] && argv[i + 1][0] != '-')
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