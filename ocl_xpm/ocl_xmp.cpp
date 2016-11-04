#include "ocl_xmp.hpp"



// miner instance
#include "opencl.h"
#include <cstdint>

#include <boost/filesystem.hpp>

// is this really needed?
//#include "uint256.h"

// hardcoded defines, looks like not working
// hardcoded defines fix this
#define RESTBITS 4
#define XINTREE
#define UNROLL
#define __OPENCL_HOST__
#include "zcash/gpu/common.h"

struct MinerInstance {
	cl_context _context;
	cl_program _program;

	cl_command_queue queue;
	clBuffer<blake2b_state> blake2bState;
	clBuffer<uint32_t> heap0;
	clBuffer<uint32_t> heap1;
	clBuffer<bsizes> nslots;
	clBuffer<proof> sols;
	clBuffer<uint32_t> numSols;
	cl_kernel _digitHKernel;
	cl_kernel _digitOKernel;
	cl_kernel _digitEKernel;
	cl_kernel _digitKKernel;
	cl_kernel _digitKernels[9];

	//hide_xmp_hack::uint256 nonce; // TODO IS THIS NEEDED????

	bool init(cl_context context, cl_program program, cl_device_id dev, unsigned threadsNum, unsigned threadsPerBlock);
};

cl_context gContext = 0;
cl_program gProgram = 0;
cl_platform_id gPlatform = 0;


bool MinerInstance::init(cl_context context,
	cl_program program,
	cl_device_id dev,
	unsigned int threadsNum,
	unsigned int threadsPerBlock)
{
	cl_int error;

	_context = context;
	_program = program;
	queue = clCreateCommandQueue(context, dev, 0, &error);

	blake2bState.init(context, 1, CL_MEM_READ_WRITE);
	heap0.init(context, sizeof(digit0) / sizeof(uint32_t), CL_MEM_HOST_NO_ACCESS);
	heap1.init(context, sizeof(digit1) / sizeof(uint32_t), CL_MEM_HOST_NO_ACCESS);
	nslots.init(context, 2, CL_MEM_READ_WRITE);
	sols.init(context, MAXSOLS, CL_MEM_READ_WRITE);
	numSols.init(context, 1, CL_MEM_READ_WRITE);

	_digitHKernel = clCreateKernel(program, "digitH", &error);
	_digitOKernel = clCreateKernel(program, "digitOdd", &error);
	_digitEKernel = clCreateKernel(program, "digitEven", &error);
	_digitKKernel = clCreateKernel(program, "digitK", &error);
	OCLR(clSetKernelArg(_digitHKernel, 0, sizeof(cl_mem), &blake2bState.DeviceData), 1);
	OCLR(clSetKernelArg(_digitHKernel, 1, sizeof(cl_mem), &heap0.DeviceData), 1);
	OCLR(clSetKernelArg(_digitHKernel, 2, sizeof(cl_mem), &nslots.DeviceData), 1);

	OCLR(clSetKernelArg(_digitOKernel, 1, sizeof(cl_mem), &heap0.DeviceData), 1);
	OCLR(clSetKernelArg(_digitOKernel, 2, sizeof(cl_mem), &heap1.DeviceData), 1);
	OCLR(clSetKernelArg(_digitOKernel, 3, sizeof(cl_mem), &nslots.DeviceData), 1);
	OCLR(clSetKernelArg(_digitEKernel, 1, sizeof(cl_mem), &heap0.DeviceData), 1);
	OCLR(clSetKernelArg(_digitEKernel, 2, sizeof(cl_mem), &heap1.DeviceData), 1);
	OCLR(clSetKernelArg(_digitEKernel, 3, sizeof(cl_mem), &nslots.DeviceData), 1);

	for (unsigned i = 1; i <= 8; i++) {
		char kernelName[32];
		sprintf(kernelName, "digit_%u", i);
		_digitKernels[i] = clCreateKernel(program, kernelName, &error);
		OCLR(clSetKernelArg(_digitKernels[i], 0, sizeof(cl_mem), &heap0.DeviceData), 1);
		OCLR(clSetKernelArg(_digitKernels[i], 1, sizeof(cl_mem), &heap1.DeviceData), 1);
		OCLR(clSetKernelArg(_digitKernels[i], 2, sizeof(cl_mem), &nslots.DeviceData), 1);
	}

	OCLR(clSetKernelArg(_digitKKernel, 0, sizeof(cl_mem), &heap0.DeviceData), 1);
	OCLR(clSetKernelArg(_digitKKernel, 1, sizeof(cl_mem), &heap1.DeviceData), 1);
	OCLR(clSetKernelArg(_digitKKernel, 2, sizeof(cl_mem), &nslots.DeviceData), 1);
	OCLR(clSetKernelArg(_digitKKernel, 3, sizeof(cl_mem), &sols.DeviceData), 1);
	OCLR(clSetKernelArg(_digitKKernel, 4, sizeof(cl_mem), &numSols.DeviceData), 1);

	return true;
}

////////////////////////////
////statics non class START

static void setheader(blake2b_state *ctx, const char *header, const uint32_t headerlen)
{
	uint32_t le_N = WN;
	uint32_t le_K = WK;
	char personal[] = "ZcashPoW01230123";
	memcpy(personal + 8, &le_N, 4);
	memcpy(personal + 12, &le_K, 4);
	blake2b_param P[1];
	P->digest_length = HASHOUT;
	P->key_length = 0;
	P->fanout = 1;
	P->depth = 1;
	P->leaf_length = 0;
	P->node_offset = 0;
	P->node_depth = 0;
	P->inner_length = 0;
	memset(P->reserved, 0, sizeof(P->reserved));
	memset(P->salt, 0, sizeof(P->salt));
	memcpy(P->personal, (const uint8_t *)personal, 16);
	blake2b_init_param(ctx, P);
	blake2b_update(ctx, (const uint8_t*)header, headerlen);
}

static void setnonce(blake2b_state *ctx, const uint8_t *nonce)
{
	blake2b_update(ctx, nonce, 32);
}

static int inline digit(cl_command_queue clQueue, cl_kernel kernel, size_t nthreads, size_t threadsPerBlock)
{
	size_t globalSize[] = { nthreads, 1, 1 };
	size_t localSize[] = { threadsPerBlock, 1 };
	OCLR(clEnqueueNDRangeKernel(clQueue, kernel, 1, 0, globalSize, localSize, 0, 0, 0), 1);
	return 0;
}


////statics non class END
////////////////////////////

ocl_xmp::ocl_xmp(int platf_id, int dev_id) { /*TODO*/
	platform_id = platf_id;
	device_id = dev_id;
	// TODO 
	threadsNum = 8192;
	wokrsize = 128; // 256;
	//threadsperblock = 128;
}

std::string ocl_xmp::getdevinfo() { /*TODO*/
	return "GPU_ID(" + std::to_string(device_id) + ")";
}

// STATICS START
int ocl_xmp::getcount() { /*TODO*/
	return 0;
}

void ocl_xmp::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version) { /*TODO*/ }

void ocl_xmp::start(ocl_xmp& device_context) {
	/*TODO*/
	device_context.is_init_success = false;
	cl_context gContext[64] = { 0 };
	cl_program gProgram[64] = { 0 };

	
	std::vector<cl_device_id> allGpus;
	if (!clInitialize(device_context.platform_id, allGpus)) {
		return;
	}
	
	// this is kinda stupid but it works
	std::vector<cl_device_id> gpus;
	for (unsigned i = 0; i < allGpus.size(); ++i) {
		if (i == device_context.device_id) {
			printf("Using device %d as GPU %d\n", i, (int)gpus.size());
			gpus.push_back(allGpus[i]);
		}
	}

	if (!gpus.size()){
		printf("Device id %d not found\n", device_context.device_id);
		return;
	}

	// context create
	for (unsigned i = 0; i < gpus.size(); i++) {
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)gPlatform, 0 };
		cl_int error;
		gContext[i] = clCreateContext(props, 1, &gpus[i], 0, 0, &error);
		//OCLR(error, false);
		if (cl_int err = error) {
			printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__);
			return;
		}
	}

	std::vector<cl_int> binstatus;
	binstatus.resize(gpus.size());

	for (size_t i = 0; i < gpus.size(); i++) {
		char kernelName[64];
		sprintf(kernelName, "equiw200k9_gpu%u.bin", (unsigned)i);
		if (!clCompileKernel(gContext[i],
			gpus[i],
			kernelName,
			{ "zcash/gpu/equihash.cl" },
			"-I./zcash/gpu -DXINTREE -DWN=200 -DWK=9 -DRESTBITS=4 -DUNROLL",
			&binstatus[i],
			&gProgram[i])) {
			return;
		}
	}

	for (unsigned i = 0; i < gpus.size(); ++i) {
		if (binstatus[i] == CL_SUCCESS) {
			device_context.context = new MinerInstance();
			if (!device_context.context->init(gContext[i], gProgram[i], gpus[i], device_context.threadsNum, device_context.wokrsize)) {
				printf("Init failed");
				return;
			}
		}
		else {
			printf("GPU %d: failed to load kernel\n", i);
			return;
		}
	}

	device_context.is_init_success = true;
}

void ocl_xmp::stop(ocl_xmp& device_context) { /*TODO*/ }

void ocl_xmp::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef,
	ocl_xmp& device_context) {
	if (device_context.is_init_success == false) {
		printf("fail OCL\n");
		//cancelf();
		return;
	}

	// move to context or somewhere or leave?
	blake2b_state initialCtx;
	setheader(&initialCtx, tequihash_header, tequihash_header_len);

	MinerInstance *miner = device_context.context;
	clFlush(miner->queue);

	/*hide_xmp_hack::uint256 nNonce = hide_xmp_hack::uint256(nonce);
	miner->nonce = nNonce;*/
	*miner->blake2bState.HostData = initialCtx;
	setnonce(miner->blake2bState.HostData, (const uint8_t*)nonce);
	memset(miner->nslots.HostData, 0, 2 * sizeof(bsizes));
	*miner->numSols.HostData = 0;
	miner->blake2bState.copyToDevice(miner->queue, false);
	miner->nslots.copyToDevice(miner->queue, false);
	miner->numSols.copyToDevice(miner->queue, false);

	digit(miner->queue, miner->_digitHKernel, device_context.threadsNum, device_context.wokrsize);
#if BUCKBITS == 16 && RESTBITS == 4 && defined XINTREE && defined(UNROLL)
	for (unsigned i = 1; i <= 8; i++)
		digit(miner->queue, miner->_digitKernels[i], device_context.threadsNum, device_context.wokrsize);
#else    
	size_t globalSize[] = { _threadsNum, 1, 1 };
	size_t localSize[] = { _threadsPerBlocksNum, 1 };
	for (unsigned r = 1; r < WK; r++) {
		if (r & 1) {
			OCL(clSetKernelArg(miner->_digitOKernel, 0, sizeof(cl_uint), &r));
			OCL(clEnqueueNDRangeKernel(miner->queue, miner->_digitOKernel, 1, 0, globalSize, localSize, 0, 0, 0));
		}
		else {
			OCL(clSetKernelArg(miner->_digitEKernel, 0, sizeof(cl_uint), &r));
			OCL(clEnqueueNDRangeKernel(miner->queue, miner->_digitEKernel, 1, 0, globalSize, localSize, 0, 0, 0));
		}
	}
#endif
	digit(miner->queue, miner->_digitKKernel, device_context.threadsNum, device_context.wokrsize);

	// get solutions
	miner->sols.copyToHost(miner->queue, true);
	miner->numSols.copyToHost(miner->queue, true);
	for (unsigned s = 0; s < miner->numSols.HostData[0]; s++)
	{
		std::vector<uint32_t> index_vector(PROOFSIZE);
		for (u32 i = 0; i < PROOFSIZE; i++) {
			index_vector[i] = miner->sols[s][i];
		}

		solutionf(index_vector, DIGITBITS, nullptr);
		if (cancelf()) return;
	}
	hashdonef();
}

// STATICS END