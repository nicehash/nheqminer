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
// fix compiler issues with standard vs2013 compiler
// file logging
// mingw compilation for windows (faster?)

int equiengine;

static ZcashStratumClient* scSig;
extern "C" void stratum_sigint_handler(int signum) 
{ 
	if (scSig) scSig->disconnect(); 
}

void print_help()
{
	std::cout << "Parameters: " << std::endl;
	std::cout << "\t-h\t\tPrint this help and quit" << std::endl;
	std::cout << "\t-l [location]\tStratum server:port" << std::endl;
	std::cout << "\t-u [username]\tUsername (pool worker)" << std::endl;
	std::cout << "\t-x [enginenum]\tEngine (-1=auto,0=tromp,1=AVX1,2=AVX2)" << std::endl;
	std::cout << "\t-p [password]\tPassword (default: x)" << std::endl;
	std::cout << "\t-t [num_thrds]\tNumber of threads (default: number of sys cores)" << std::endl;
	std::cout << "\t-d [level]\tDebug print level (0 = print all, 5 = fatal only, default: 2)" << std::endl;
	std::cout << "\t-b [hashes]\tRun in benchmark mode (default: 100 hashes)" << std::endl;
	std::cout << "\t-a [port]\tLocal API port (default: 0 = do not bind)" << std::endl;
	std::cout << std::endl;
}

#ifdef _MSC_VER
void init_logging(boost::log::core_ptr cptr, int level);
#else
#include <iostream>

#include <boost/log/core/core.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

//namespace logging = boost::log;
//namespace keywords = boost::log::keywords;
namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace src = boost::log::sources;
//namespace fmt = boost::log::formatters;
//namespace flt = boost::log::filters;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;
#endif


int main(int argc, char* argv[])
{
	std::cout << "Kost CPU Miner - https://github.com/kost/nheqminer " STANDALONE_MINER_NAME "/" STANDALONE_MINER_VERSION << std::endl;
	std::cout << "Thanks to Zcash developers and community, nicehash, tromp and xenoncat. Donate!" << std::endl;
	std::cout << "BTC:1KHRiwNdFiL4uFUGFEpbG7t2F3pUcttLuX ZEC:t1JBZzdaUUSJDs8q7SUxcCSzakThqtNRtNv" << std::endl;
	std::cout << std::endl;

	std::string location = "eu1-zcash.flypool.org:3333";
	std::string user = "t1JBZzdaUUSJDs8q7SUxcCSzakThqtNRtNv";
	std::string password = "x";
	int num_threads = -1;
	bool benchmark = false;
	int log_level = 2;
	int num_hashes = 100;
	int api_port = 0;

	// set defaults
	equiengine = -1;

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
		case 'x':
			equiengine = atoi(argv[++i]);
			break;
		case 'a':
			api_port = atoi(argv[++i]);
			break;
		}
	}

#ifdef _MSC_VER
    init_logging(boost::log::core::get(), log_level);
#else
    std::cout << "Setting log level to " << log_level << std::endl;
    boost::log::add_console_log(
        std::clog,
        boost::log::keywords::auto_flush = true,
        boost::log::keywords::filter = boost::log::trivial::severity >= log_level,
        boost::log::keywords::format = (
        boost::log::expressions::stream
            << "[" << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%H:%M:%S")
            << "][" << boost::log::expressions::attr<boost::log::attributes::current_thread_id::value_type>("ThreadID")
            << "] "  << boost::log::expressions::smessage
        )
    );
    boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
    boost::log::core::get()->add_global_attribute("ThreadID", boost::log::attributes::current_thread_id());
#endif



	if (!benchmark)
	{
		size_t delim = location.find(':');
		std::string host = location.substr(0, delim);
		std::string port = location.substr(delim+1);

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

		miner.onSolutionFound([&](const EquihashSolution& solution, const std::string& jobid) {
			return sc.submit(&solution, jobid);
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
					speed.GetSolutionSpeed() << " Sol/s" << 
					//accepted << " AS/min, " << 
					//(allshares - accepted) << " RS/min" 
					CL_N;
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

