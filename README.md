# Build instructions:

### Dependencies:
  - Boost 1.54+

## Windows:

Windows builds made by us are available here: https://github.com/nicehash/nheqminer/releases

Download and install:
- [AMD APP SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/) (if not needed remove **USE_OCL_XMP** from **nheqminer** Preprocessor definitions under Properties > C/C++ > Preprocessor)
- [CUDA SDK](https://developer.nvidia.com/cuda-downloads) (if not needed remove **USE_CUDA_TROMP** from **nheqminer** Preprocessor definitions under Properties > C/C++ > Preprocessor)
- Visual Studio 2013 Community: https://www.visualstudio.com/en-us/news/releasenotes/vs2013-community-vs
- Visual Studio Update 5 installed
- 64 bit version only

Open **nheqminer.sln** under **nheqminer/nheqminer.sln** and build. You will have to build ReleaseSSE2 cpu_tromp project first, then Release7.5 cuda_tromp project, then select Release and build all.


## Linux

Work in progress.

Working solvers CPU_TROMP, CPU_XENONCAT, CUDA_TROMP, OCL_XMP, OCL_SILENTARMY

## Linux (Ubuntu 14.04 / 16.04) Build CPU_XENONCAT:

 - Open terminal and run the following commands:
   - `sudo apt-get install cmake build-essential libboost-all-dev`
   - `git clone -b Linux https://github.com/nicehash/nheqminer.git`
   - `cd nheqminer/cpu_xenoncat/Linux/asm/`
   - `sh assemble.sh`
   - `cd ../../../Linux_cmake/nheqminer_cpu_xenoncat`
   - `cmake .`
   - `make -j $(nproc)`

## Linux (Ubuntu 14.04 / 16.04) Build CUDA_TROMP:

 - Open terminal and run the following commands:
   - **Ubuntu 14.04**:
     - `wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb`
     - `sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb`
   - **Ubuntu 16.04**:
     - `wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb`
     - `sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb`
   - `sudo apt-get update`
   - `sudo apt-get install cuda`
   - `sudo apt-get install cuda-toolkit-8-0`
   - `sudo apt-get install cmake build-essential libboost-all-dev`
   - `git clone -b Linux https://github.com/nicehash/nheqminer.git`
   - `cd nheqminer/Linux_cmake/nheqminer_cuda_tromp && cmake . && make -j $(nproc)`
   - or specify your compute version for example 50 like so `cd nheqminer/Linux_cmake/nheqminer_cuda_tromp && cmake COMPUTE=50 . && make`

## Linux (16.04) Build OCL_XMP, OCL_SILENTARMY:

 - Open terminal and run the following commands:
   - [AMD APP SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
   - and make sure you have the [AMD drivers](http://support.amd.com/en-us/download) installed
   - install them to the default paths
   - `sudo apt-get install mesa-common-dev`
   - `sudo apt-get install cmake build-essential libboost-all-dev`
   - `git clone -b Linux https://github.com/nicehash/nheqminer.git`
   - `cd nheqminer/Linux_cmake/nheqminer_AMD && cmake . -DOPENCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so -DOPENCL_INCLUDE_DIRECTORY=/opt/AMDAPPSDK-3.0/include && make -j $(nproc)`
   - `cp ../../3rdparty/amd_bins_linux/* -r .`
   - `cp ../../3rdparty/amd_silentarmy_kernels/* -r .`

   

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
