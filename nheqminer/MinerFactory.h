#pragma once

#include <AvailableSolvers.h>

class MinerFactory
{
public:
	MinerFactory(bool use_xenoncat, bool use_cuda_djezo, bool use_silentarmy)
		: _use_xenoncat(use_xenoncat), _use_cuda_djezo(use_cuda_djezo), _use_silentarmy(use_silentarmy) {
	}

	~MinerFactory();

	std::vector<ISolver *> GenerateSolvers(int cpu_threads, int cuda_count, int* cuda_en, int* cuda_b, int* cuda_t,
		int opencl_count, int opencl_platf, int* opencl_en, int* opencl_t);
	void ClearAllSolvers();

private:
	std::vector<ISolver *> _solvers;

	bool _use_xenoncat = true;
	bool _use_cuda_djezo = true;
	bool _use_silentarmy = true;

	ISolver * GenCPUSolver(int use_opt);
	ISolver * GenCUDASolver(int dev_id, int blocks, int threadsperblock);
	ISolver * GenOPENCLSolver(int platf_id, int dev_id);

};

