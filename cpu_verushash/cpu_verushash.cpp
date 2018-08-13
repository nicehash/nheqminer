// (C) 2018 Michael Toutonghi
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <iostream>
#include <functional>
#include <vector>
#include <stdint.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <string.h>
#include <stdlib.h>
#endif

#include "cpu_verushash.hpp"
#include "primitives/block.h"

void cpu_verushash::start(cpu_verushash& device_context) 
{
	device_context.pVHW = new CVerusHashWriter(SER_GETHASH, PROTOCOL_VERSION);
}

void cpu_verushash::stop(cpu_verushash& device_context) 
{ 
	delete device_context.pVHW;
}

void cpu_verushash::solve_verus(CBlockHeader &bh, 
	arith_uint256 &target,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef,
	cpu_verushash &device_context)
{
	CVerusHashWriter &vhw = *(device_context.pVHW);
	CVerusHash &vh = vhw.GetState();
	uint256 curHash;

	// prepare the hash state
	vhw.Reset();
	vhw << bh;

	// clear any extra data
	vh.ClearExtra();

	// loop the requested number of times or until canceled. determine if we 
	// found a winner, and send all winners found as solutions. count only one hash. 
	// hashrate is determined by multiplying hash by VERUSHASHES_PER_SOLVE, with VerusHash, only
	// hashrate and sharerate are valid, solutionrate will equal sharerate
	for (int64_t i; i < VERUSHASHES_PER_SOLVE; i++)
	{
		*(vh.ExtraI64Ptr()) = i;
		vh.ExtraHash((unsigned char *)&curHash);
		if (UintToArith256(curHash) > target)
			continue;

		int extraSpace = (bh.nSolution.size() % 32) - 2;
		*((int64_t *)&(bh.nSolution.data()[bh.nSolution.size() - extraSpace])) = i;

		solutionf(std::vector<uint32_t>(0), bh.nSolution.size(), bh.nSolution.data());
		if (cancelf()) return;
	}

	hashdonef();
}