# Build instructions:

### Dependencies:
  - Boost 1.62+


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
  - Currently support only static building (CPU_XENONCAT, CPU_TROMP are enabled by default, check **CMakeLists.txt** in **nheqminer** root folder)
  -
  - If not on Ubuntu make sure you have **fasm** installed and accessible in **PATH**
  - After that open the terminal and run the following script:
    
```bash    
#!/bin/bash
#
sudo apt-get install build-essential g++ python-dev autotools-dev libicu-dev build-essential libbz2-dev libboost-all-dev fasm
git clone https://github.com/nicehash/nheqminer.git
cd ./nheqminer && git pull && cd ../
#
rm -r ./build
cd nheqminer/cpu_xenoncat/asm_linux/
rm fasm
cp /usr/bin/fasm ./
chmod 755 assemble.sh fasm
sh assemble.sh
cd ../../../
mkdir build && cd build
cmake ../nheqminer
make -j $(nproc)
#
exit 0
```
    
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
