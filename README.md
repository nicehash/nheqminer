# Build instructions:

## Windows:

Windows builds made by us are available here: https://github.com/feeleep75/nheqminer/releases

Download and install:
- Visual Studio 2013 Community: https://www.visualstudio.com/en-us/news/releasenotes/vs2013-community-vs
- Visual C++ Compiler November 2013 CTP: https://www.microsoft.com/en-us/download/details.aspx?id=41151

Open **nheqminer.sln** under **nheqminer/nheqminer.sln** and build.

## Linux cmake **recommended** (Tested on Ubuntu Desktop 14.04 and 16.04 and Ubuntu server 14.04):
You should have **CMake** installed (2.8 minimal version), boost (install from the repositories or download boost manually build and install it manually), download the sources manually or via git. 
Under Ubuntu open a terminal and run the following commands:
  - `sudo apt-get install cmake build-essential libboost-all-dev`
  - `git clone https://github.com/nicehash/nheqminer.git`
  - `cd nheqminer/nheqminer`
  - `mkdir build`
  - `cd build`
  - `cmake ..`
  - `make`


## Linux (Ubuntu/Debian based, Tested on Ubuntu 16.04):
To build under Ubuntu Linux make sure you have Qt5 installed. You can install it manually from [Qt website](https://www.qt.io/) or install it from the command line: `sudo apt-get install qt5-default`.
Open a terminal and cd to nheqminer root folder and run the following commands (make sure you have qmake in your PATH, if installed manually from Qt website you will have to export it to your PATH):
  - `git clone https://github.com/feeleep75/nheqminer.git`
  - `cd nheqminer`
  - `mkdir build`
  - `cd build`
  - `qmake ../nheqminer/nheqminer.pro`
  - `make`
  

# Run instructions:

If run without parameters, miner will start mining with 75% of available virtual cores on NiceHash. Use parameter -h to learn about available parameters:

        -h              Print this help and quit
        -l [location]   Stratum server (zec.coinmine.pl:7007)
        -u [username]   Username (worker_name)
        -p [password]   Password (default: x)
        -t [num_thrds]  Number of threads (default: number of sys cores)
        -d [level]      Debug print level (0 = print all, 5 = fatal only, default: 2)
        -b [hashes]     Run in benchmark mode (default: 100 hashes)
        -a [port]       Local API port (default: 0 = do not bind)
        
Example to run benchmark:

        nheqminer.exe -b
        
Example to run with full logging (including network dump):

        nheqminer.exe -d 0
        
Example to mine:

        nheqmine.exe -l zec.coinmine.pl:7007 -u feeleep.1 -p x -t 6

<i>Note: if you have a 4-core CPU with hyper threading enabled (total 8 threads) it is best to run with only 6 threads (experimental benchmarks shows that best results are achieved with 75% threads utilized)</i>
