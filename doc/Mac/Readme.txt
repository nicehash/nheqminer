VerusCoin nheqminer v0.6.3-beta
Using VerusHash v0.3.13-beta

This software needs to be run on a terminal. To find it, open the Applications folder, then open the Utilities folder and finally open the Terminal application. The terminal can also be found using spotlight and searching for “terminal”.
Once the terminal launches, navigate to the nheqminer directory by running:

cd ~/Downloads/nheqminer

Now you can run the nheqminer by running:
 ./nheqminer [Parameters]

Parameters:
        -h              Print this help and quit
        -l [location]   Stratum server:port
        -u [username]   Username (bitcoinaddress)
        -p [passwd]     password
        -a [port]       Local API port (default: 0 = do not bind)
        -d [level]      Debug print level (0 = print all, 5 = fatal only, default: 2)
        -b [hashes]     Run in benchmark mode (default: 200 iterations)

VerusHash settings
        -v              Mine with VerusHash algorithm
        -vm [magicnum]  set magic number for VerusHash chain other than VRSC

CPU settings
        -t [num_thrds]  Number of CPU threads
        -e [ext]        Force CPU ext (0 = SSE2, 1 = AVX, 2 = AVX2)

NVIDIA CUDA settings
        -ci             CUDA info
        -cv [ver]       Set CUDA solver (0 = djeZo, 1 = tromp)
        -cd [devices]   Enable CUDA mining on spec. devices
        -cb [blocks]    Number of blocks
        -ct [tpb]       Number of threads per block
Example: -cd 0 2 -cb 12 16 -ct 64 128


To mine verus, use the -v flag to mine with the VerusHash algorithm. Use parameter -h to learn about available parameters:

Example to run benchmark on your CPU:

        ./nheqminer -v -b

Example to mine on your CPU with your own VRSC address and worker1 on Stratum USA server:

        ./nheqminer -v -l us-veruscoin.miningpools.cloud:2052 -u YOUR_VRSC_ADDRESS_HERE.worker1

Example to mine on your CPU with your own VRSC address and worker1 on Stratum Asia server, using 6 threads:

        ./nheqminer -v -l asia-veruscoin.miningpools.cloud:2052 -u YOUR_VRSC_ADDRESS_HERE.worker1 -t 6

Example to mine on your CPU with your own VRSC address and worker1 on Stratum EU server, using all threads:

        ./nheqminer -v -l veruscoin.miningpools.cloud:2052 -u YOUR_VRSC_ADDRESS_HERE.worker1 -t $(sysctl -n hw.physicalcpu)
