# Build instructions:

### Dependencies:
  - Boost 1.62+

## Windows:

Windows builds made by us are available here: https://github.com/nicehash/nheqminer/releases

Download and install:
- [CUDA SDK](https://developer.nvidia.com/cuda-downloads) (if not needed remove **USE_CUDA_TROMP** and **USE_CUDA_DJEZO** from **nheqminer** Preprocessor definitions under Properties > C/C++ > Preprocessor)
- Visual Studio 2013 Community: https://www.visualstudio.com/en-us/news/releasenotes/vs2013-community-vs
- [Visual Studio Update 5](https://www.microsoft.com/en-us/download/details.aspx?id=48129) installed
- 64 bit version only

Open **nheqminer.sln** under **nheqminer/nheqminer.sln** and build. You will have to build ReleaseSSE2 cpu_tromp project first, then Release7.5 cuda_tromp project, then select Release and build all.

### Enabled solvers: 
  - USE_CPU_TROMP
  - USE_CPU_XENONCAT
  - USE_CUDA_TROMP
  - USE_CUDA_DJEZO

If you don't wan't to build with all solvlers you can go to **nheqminer Properties > C/C++ > Preprocessor > Preprocessor Definitions** and remove the solver you don't need.

## Linux
Work in progress.
Working solvers CPU_TROMP, CPU_XENONCAT, CUDA_TROMP, CUDA_DJEZO

### General instructions:
  - Install CUDA SDK v8 (make sure you have cuda libraries in **LD_LIBRARY_PATH** and cuda toolkit bins in **PATH**)
    - example on Ubuntu:
    - LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib64/stubs"
    - PATH="$PATH:/usr/local/cuda-8.0/"
    - PATH="$PATH:/usr/local/cuda-8.0/bin"

  - Use Boost 1.62+ (if it is not available from the repos you will have to download and build it yourself)
  - CMake v3.5 (if it is not available from the repos you will have to download and build it yourself)
  - Currently support only static building (CPU_XENONCAT, CUDA_DJEZO are enabled by default, check **CMakeLists.txt** in **nheqminer** root folder)
  - If not on Ubuntu make sure you have **fasm** installed and accessible in **PATH**
  - After that open the terminal and run the following commands:
    - `git clone https://github.com/nicehash/nheqminer.git`
    - Generating asm object file:
      - **On Ubuntu**:
        - `cd nheqminer/cpu_xenoncat/asm_linux/`
        - `sh assemble.sh`
      - **bundeled fasm not compatible**:
        - delete/replace (inside **nheqminer/cpu_xenoncat/asm_linux/** directory) with fasm binary compatible with your distro
        - `cd nheqminer/cpu_xenoncat/asm_linux/`
        - `sh assemble.sh`
    - `cd ../../../`
    - `mkdir build && cd build`
    - `cmake ../nheqminer`
    - `make -j $(nproc)`
    
# Run instructions:

Parameters: 
	-h		Print this help and quit
	-l [location]	Stratum server:port
	-u [username]	Username (bitcoinaddress)
	-a [port]	Local API port (default: 0 = do not bind)
	-d [level]	Debug print level (0 = print all, 5 = fatal only, default: 2)
	-b [hashes]	Run in benchmark mode (default: 200 iterations)

CPU settings
	-t [num_thrds]	Number of CPU threads
	-e [ext]	Force CPU ext (0 = SSE2, 1 = AVX, 2 = AVX2)

NVIDIA CUDA settings
	-ci		CUDA info
	-cd [devices]	Enable CUDA mining on spec. devices
	-cb [blocks]	Number of blocks
	-ct [tpb]	Number of threads per block
Example: -cd 0 2 -cb 12 16 -ct 64 128

If run without parameters, miner will start mining with 75% of available logical CPU cores. Use parameter -h to learn about available parameters:

Example to run benchmark on your CPU:

        nheqminer -b
        
Example to mine on your CPU with your own BTC address and worker1 on NiceHash USA server:

        nheqminer -l equihash.usa.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1

Example to mine on your CPU with your own BTC address and worker1 on EU server, using 6 threads:

        nheqminer -l equihash.eu.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1 -t 6

<i>Note: if you have a 4-core CPU with hyper threading enabled (total 8 threads) it is best to run with only 6 threads (experimental benchmarks shows that best results are achieved with 75% threads utilized)</i>

Example to mine on your CPU as well on your CUDA GPUs with your own BTC address and worker1 on EU server, using 6 CPU threads and 2 CUDA GPUs:

        nheqminer -l equihash.eu.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1 -t 6 -cd 0 1
