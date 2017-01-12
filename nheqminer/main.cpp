#include <iostream>

#include "version.h"
#include "arith_uint256.h"
#include "primitives/block.h"
#include "streams.h"

#include "MinerFactory.h"

#include "libstratum/StratumClient.h"

#if defined(USE_OCL_XMP) || defined(USE_OCL_SILENTARMY)
#include "../ocl_device_utils/ocl_device_utils.h"
#define PRINT_OCL_INFO
#endif

#include <thread>
#include <chrono>
#include <atomic>
#include <bitset>

#include "speed.hpp"
#include "api.hpp"

#include <boost/log/core/core.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace src = boost::log::sources;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;

#ifdef __linux__
#define __cpuid(out, infoType)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType));
#define __cpuidex(out, infoType, ecx)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType), "c" (ecx));
#endif

// TODO:
// #1 file logging
// #2 mingw compilation for windows (faster?)
// #3 benchmark accuracy fix: first wait for solvers to init and then measure speed
// #4 Linux fix cmake to generate all in one binary (just like Windows)
// #5 after #4 is done add solver chooser for CPU and CUDA devices (general and per device), example: [-s 0 automatic, -s 1 solver1, -s 2 solver2, ...]

int use_avx = 0;
int use_avx2 = 0;
int use_old_cuda = 0;
int use_old_xmp = 0;

// TODO move somwhere else
MinerFactory *_MinerFactory = nullptr;

// stratum client sig
static ZcashStratumClient* scSig = nullptr;

extern "C" void stratum_sigint_handler(int signum) 
{ 
	if (scSig) scSig->disconnect();
	if (_MinerFactory) _MinerFactory->ClearAllSolvers();
}

void print_help()
{
	std::cout << "Parameters: " << std::endl;
	std::cout << "\t-h\t\tPrint this help and quit" << std::endl;
#ifndef ZCASH_POOL
	std::cout << "\t-l [location]\tStratum server:port" << std::endl;
	std::cout << "\t-u [username]\tUsername (bitcoinaddress)" << std::endl;
#else
	std::cout << "\t-l [location]\tLocation (eu, usa)" << std::endl;
	std::cout << "\t-u [username]\tUsername (Zcash wallet address)" << std::endl;
#endif
	std::cout << "\t-a [port]\tLocal API port (default: 0 = do not bind)" << std::endl;
	std::cout << "\t-d [level]\tDebug print level (0 = print all, 5 = fatal only, default: 2)" << std::endl;
	std::cout << "\t-b [hashes]\tRun in benchmark mode (default: 200 iterations)" << std::endl;
	std::cout << std::endl;
	std::cout << "CPU settings" << std::endl;
	std::cout << "\t-t [num_thrds]\tNumber of CPU threads" << std::endl;
	std::cout << "\t-e [ext]\tForce CPU ext (0 = SSE2, 1 = AVX, 2 = AVX2)" << std::endl;
	std::cout << std::endl;
	std::cout << "NVIDIA CUDA settings" << std::endl;
	std::cout << "\t-ci\t\tCUDA info" << std::endl;
	std::cout << "\t-cv [ver]\tSet CUDA solver (0 = djeZo, 1 = tromp)" << std::endl;
	std::cout << "\t-cd [devices]\tEnable CUDA mining on spec. devices" << std::endl;
	std::cout << "\t-cb [blocks]\tNumber of blocks" << std::endl;
	std::cout << "\t-ct [tpb]\tNumber of threads per block" << std::endl;
	std::cout << "Example: -cd 0 2 -cb 12 16 -ct 64 128" << std::endl;
	std::cout << std::endl;
	//std::cout << "OpenCL settings" << std::endl;
	//std::cout << "\t-oi\t\tOpenCL info" << std::endl;
	//std::cout << "\t-ov [ver]\tSet OpenCL solver (0 = silentarmy, 1 = xmp)" << std::endl;
	//std::cout << "\t-op [platf]\tSet OpenCL platform to selecd platform devices (-od)" << std::endl;
	//std::cout << "\t-od [devices]\tEnable OpenCL mining on spec. devices (specify plafrom number first -op)" << std::endl;
	//std::cout << "\t-ot [threads]\tSet number of threads per device" << std::endl;
	////std::cout << "\t-cb [blocks]\tNumber of blocks" << std::endl;
	////std::cout << "\t-ct [tpb]\tNumber of threads per block" << std::endl;
	//std::cout << "Example: -op 2 -od 0 2" << std::endl; //-cb 12 16 -ct 64 128" << std::endl;
	std::cout << std::endl;
}


void print_cuda_info()
{
#if defined(USE_CUDA_DJEZO) || defined(USE_CUDA_TROMP)
#ifdef USE_CUDA_DJEZO
    int num_devices = cuda_djezo::getcount();
#elif USE_CUDA_TROMP
    int num_devices = cuda_tromp::getcount();
#endif

	std::cout << "Number of CUDA devices found: " << num_devices << std::endl;

	for (int i = 0; i < num_devices; ++i)
	{
		std::string gpuname, version;
		int smcount;
#ifdef USE_CUDA_DJEZO
        cuda_djezo::getinfo(0, i, gpuname, smcount, version);
#elif USE_CUDA_TROMP
        cuda_tromp::getinfo(0, i, gpuname, smcount, version);
#endif
		std::cout << "\t#" << i << " " << gpuname << " | SM version: " << version << " | SM count: " << smcount << std::endl;
	}
#endif
}

void print_opencl_info() {
#ifdef PRINT_OCL_INFO
	ocl_device_utils::print_opencl_devices();
#endif
}

#define MAX_INSTANCES 8 * 2

int cuda_enabled[MAX_INSTANCES] = { 0 };
int cuda_blocks[MAX_INSTANCES] = { 0 };
int cuda_tpb[MAX_INSTANCES] = { 0 };

int opencl_enabled[MAX_INSTANCES] = { 0 };
int opencl_threads[MAX_INSTANCES] = { 0 };
// todo: opencl local and global worksize


void detect_AVX_and_AVX2()
{
    // Fix on Linux
	//int cpuInfo[4] = {-1};
	std::array<int, 4> cpui;
	std::vector<std::array<int, 4>> data_;
	std::bitset<32> f_1_ECX_;
	std::bitset<32> f_7_EBX_;

	// Calling __cpuid with 0x0 as the function_id argument
	// gets the number of the highest valid function ID.
	__cpuid(cpui.data(), 0);
	int nIds_ = cpui[0];

	for (int i = 0; i <= nIds_; ++i)
	{
		__cpuidex(cpui.data(), i, 0);
		data_.push_back(cpui);
	}

	if (nIds_ >= 1)
	{
		f_1_ECX_ = data_[1][2];
		use_avx = f_1_ECX_[28];
	}

	// load bitset with flags for function 0x00000007
	if (nIds_ >= 7)
	{
		f_7_EBX_ = data_[7][1];
		use_avx2 = f_7_EBX_[5];
	}
}


void start_mining(int api_port, const std::string& host, const std::string& port,
	const std::string& user, const std::string& password,
	ZcashStratumClient* handler, const std::vector<ISolver *> &i_solvers)
{
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
	
	ZcashMiner miner(i_solvers);
	ZcashStratumClient sc{
		io_service, &miner, host, port, user, password, 0, 0
	};

	miner.onSolutionFound([&](const EquihashSolution& solution, const std::string& jobid) {
		return sc.submit(&solution, jobid);
	});

	handler = &sc;
	signal(SIGINT, stratum_sigint_handler);

	int c = 0;
	while (sc.isRunning()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (++c % 1000 == 0)
		{
			double allshares = speed.GetShareSpeed() * 60;
			double accepted = speed.GetShareOKSpeed() * 60;
			BOOST_LOG_TRIVIAL(info) << CL_YLW "Speed [" << INTERVAL_SECONDS << " sec]: " <<
				speed.GetHashSpeed() << " I/s, " <<
				speed.GetSolutionSpeed() << " Sols/s" <<
				//accepted << " AS/min, " << 
				//(allshares - accepted) << " RS/min" 
				CL_N;
		}
		if (api) while (api->poll()) {}
	}

	if (api) delete api;
}


int main(int argc, char* argv[])
{
#if defined(WIN32) && defined(NDEBUG)
	system(""); // windows 10 colored console
#endif

	std::cout << std::endl;
	std::cout << "\t==================== www.nicehash.com ====================" << std::endl;
	std::cout << "\t\tEquihash CPU&GPU Miner for NiceHash v" STANDALONE_MINER_VERSION << std::endl;
	std::cout << "\tThanks to Zcash developers for providing base of the code." << std::endl;
	std::cout << "\t    Special thanks to tromp, xenoncat and djeZo for providing "<< std::endl;
	std::cout << "\t      optimized CPU and CUDA equihash solvers." << std::endl;
	std::cout << "\t==================== www.nicehash.com ====================" << std::endl;
	std::cout << std::endl;

	std::string location = "equihash.eu.nicehash.com:3357";
	std::string user = "34HKWdzLxWBduUfJE9JxaFhoXnfC6gmePG";
	std::string password = "x";
	int num_threads = 0;
	bool benchmark = false;
	int log_level = 2;
	int num_hashes = 200;
	int api_port = 0;
	int cuda_device_count = 0;
	int cuda_bc = 0;
	int cuda_tbpc = 0;
	int opencl_platform = 0;
	int opencl_device_count = 0;
	int force_cpu_ext = -1;
	int opencl_t = 0;

	for (int i = 1; i < argc; ++i)
	{
		if (argv[i][0] != '-') continue;

		switch (argv[i][1])
		{
		case 'c':
		{
			switch (argv[i][2])
			{
			case 'i':
				print_cuda_info();
				return 0;
			case 'v':
				use_old_cuda = atoi(argv[++i]);
				break;
			case 'd':
				while (cuda_device_count < MAX_INSTANCES && i + 1 < argc)
				{
					try
					{
						cuda_enabled[cuda_device_count] = std::stol(argv[++i]);
						++cuda_device_count;
					}
					catch (...)
					{
						--i;
						break;
					}
				}
				break;
			case 'b':
				while (cuda_bc < MAX_INSTANCES && i + 1 < argc)
				{
					try
					{
						cuda_blocks[cuda_bc] = std::stol(argv[++i]);
						++cuda_bc;
					}
					catch (...)
					{
						--i;
						break;
					}
				}
				break;
			case 't':
				while (cuda_tbpc < MAX_INSTANCES && i + 1 < argc)
				{
					try
					{
						cuda_tpb[cuda_tbpc] = std::stol(argv[++i]);
						++cuda_tbpc;
					}
					catch (...)
					{
						--i;
						break;
					}
				}
				break;
			}
			break;
		}
		//case 'o':
		//{
		//	switch (argv[i][2])
		//	{
		//	case 'i':
		//		print_opencl_info();
		//		return 0;
		//	case 'v':
		//		use_old_xmp = atoi(argv[++i]);
		//		break;
		//	case 'p':
		//		opencl_platform = std::stol(argv[++i]);
		//		break;
		//	case 'd':
		//		while (opencl_device_count < 8 && i + 1 < argc)
		//		{
		//			try
		//			{
		//				opencl_enabled[opencl_device_count] = std::stol(argv[++i]);
		//				++opencl_device_count;
		//			}
		//			catch (...)
		//			{
		//				--i;
		//				break;
		//			}
		//		}
		//		break;
		//	case 't':
		//		while (opencl_t < 8 && i + 1 < argc)
		//		{
		//			try
		//			{
		//				opencl_threads[opencl_t] = std::stol(argv[++i]);
		//				++opencl_t;
		//			}
		//			catch (...)
		//			{
		//				--i;
		//				break;
		//			}
		//		}
		//		break;
		//		// TODO extra parameters for OpenCL
		//	}
		//	break;
		//}
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
		case 'e':
			force_cpu_ext = atoi(argv[++i]);
			break;
		}
	}

	if (force_cpu_ext >= 0)
	{
		switch (force_cpu_ext)
		{
		case 1:
			use_avx = 1;
			break;
		case 2:
			use_avx = 1;
			use_avx2 = 1;
			break;
		}
	}
	else
		detect_AVX_and_AVX2();

	// init_logging init START
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
	// init_logging init END

	BOOST_LOG_TRIVIAL(info) << "Using SSE2: YES";
	BOOST_LOG_TRIVIAL(info) << "Using AVX: " << (use_avx ? "YES" : "NO");
	BOOST_LOG_TRIVIAL(info) << "Using AVX2: " << (use_avx2 ? "YES" : "NO");

	try
	{
		_MinerFactory = new MinerFactory(use_avx == 1, use_old_cuda == 0, use_old_xmp == 0);
		if (!benchmark)
		{
			if (user.length() == 0)
			{
				BOOST_LOG_TRIVIAL(error) << "Invalid address. Use -u to specify your address.";
				return 0;
			}

			size_t delim = location.find(':');
			std::string host = delim != std::string::npos ? location.substr(0, delim) : location;
			std::string port = delim != std::string::npos ? location.substr(delim + 1) : "2142";

			start_mining(api_port, host, port, user, password,
				scSig,
				_MinerFactory->GenerateSolvers(num_threads, cuda_device_count, cuda_enabled, cuda_blocks,
				cuda_tpb, opencl_device_count, opencl_platform, opencl_enabled, opencl_threads));
		}
		else
		{
			Solvers_doBenchmark(num_hashes, _MinerFactory->GenerateSolvers(num_threads, cuda_device_count, cuda_enabled, cuda_blocks, cuda_tpb, opencl_device_count, opencl_platform, opencl_enabled, opencl_threads));
		}
	}
	catch (std::runtime_error& er)
	{
		BOOST_LOG_TRIVIAL(error) << er.what();
	}

	boost::log::core::get()->remove_all_sinks();

	return 0;
}

