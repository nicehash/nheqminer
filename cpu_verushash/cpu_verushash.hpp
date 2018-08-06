// (C) 2018 Michael Toutonghi
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
#pragma once 

#include "primitives/block.h"
#include "../nheqminer/hash.h"

struct cpu_verushash
{
	std::string getdevinfo() { return ""; }

	static void start(cpu_verushash& device_context);

	static void stop(cpu_verushash& device_context);

	static void solve_verus(CBlockHeader &bh, 
		arith_uint256 &target,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		cpu_verushash &device_context);

	std::string getname()
	{ 
		return "CPU-VERUSHASH-AES";
	}

	CVerusHashWriter *pVHW;
	int use_opt; // unused
};

