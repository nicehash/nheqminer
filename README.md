# Build instructions:
### Dependencies:
  - Boost 1.54+
## Windows:

Windows builds made by us are available here: https://github.com/nicehash/nheqminer/releases

Download and install:
- [AMD APP SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/) (if not needed remove **USE_OCL_XMP** from **nheqminer** Preprocessor definitions under Properties > C/C++ > Preprocessor)
- [CUDA SDK](https://developer.nvidia.com/cuda-downloads) (if not needed remove **USE_CUDA_TROMP** from **nheqminer** Preprocessor definitions under Properties > C/C++ > Preprocessor)
- Visual Studio 2013 Community: https://www.visualstudio.com/en-us/news/releasenotes/vs2013-community-vs
- Visual C++ Compiler November 2013 CTP: https://www.microsoft.com/en-us/download/details.aspx?id=41151

Open **nheqminer.sln** under **nheqminer/nheqminer.sln** and build.


## Linux
Work in progress.

Working solvers CPU_TROMP, CPU_XENONCAT, CUDA_TROMP

Work in progress (OCL_XMP)
## Linux (Ubuntu 14.04 / 16.04) Build  CPU_XENONCAT:
 - Open terminal and run the following commands:
   - `sudo apt-get install cmake build-essential libboost-all-dev`
   - `git clone -b Linux https://github.com/nicehash/nheqminer.git`
   - `cd nheqminer/cpu_xenoncat/Linux/asm/ && sh assemble.sh && cd ../../../Linux_cmake/nheqminer_cpu && cmake . && make`

 ## Linux (Ubuntu 14.04 / 16.04) Build  CUDA_TROMP:
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
   - `cd nheqminer/Linux_cmake/nheqminer_cuda_tromp && cmake . && make`
   - or specify your compute version for example 50 like so `cd nheqminer/Linux_cmake/nheqminer_cuda_tromp && cmake COMPUTE=50 . && make`

# Run instructions:

If run without parameters, miner will start mining with 75% of available virtual cores on NiceHash. Use parameter -h to learn about available parameters:

        -h              Print this help and quit
        -l [location]   Location (Stratum server:port)
        -u [username]   Username (bitcoinaddress)
        -p [password]   Password (default: x)
        -t [num_thrds]  Number of threads (default: number of sys cores)
        -d [level]      Debug print level (0 = print all, 5 = fatal only, default: 2)
        -b [hashes]     Run in benchmark mode (default: 100 hashes)
        -a [port]       Local API port (default: 0 = do not bind)
        
Example to run benchmark:

        nheqminer_x64_AVX.exe -b
        
Example to run with full logging (including network dump):

        nheqminer_x64_AVX.exe -d 0
        
Example to mine with your own BTC address and worker1 on USA server:

        nheqminer_x64_AVX.exe -l equihash.usa.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1

Example to mine with your own BTC address and worker1 on EU server, using 6 threads:

        nheqminer_x64_AVX.exe -l equihash.eu.nicehash.com:3357 -u YOUR_BTC_ADDRESS_HERE.worker1 -t 6

<i>Note: if you have a 4-core CPU with hyper threading enabled (total 8 threads) it is best to run with only 6 threads (experimental benchmarks shows that best results are achieved with 75% threads utilized)</i>
