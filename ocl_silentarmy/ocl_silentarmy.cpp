#include "ocl_silentarmy.hpp"

//#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <sys/types.h>
//#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <unistd.h>
//#include <getopt.h>
#include <errno.h>


#include "opencl.h"

#include <fstream>

#include "sa_blake.h"

typedef uint8_t		uchar;
typedef uint32_t	uint;
typedef uint64_t	ulong;
#include "param.h"

#define MIN(A, B)	(((A) < (B)) ? (A) : (B))
#define MAX(A, B)	(((A) > (B)) ? (A) : (B))

#define WN PARAM_N
#define WK PARAM_K

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))

#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))

typedef struct  debug_s
{
	uint32_t    dropped_coll;
	uint32_t    dropped_stor;
}               debug_t;

struct OclContext {
	cl_context _context;
	cl_program _program;
	cl_device_id _dev_id;

	cl_platform_id platform_id = 0;

	cl_command_queue queue;

	cl_kernel k_init_ht;
	cl_kernel k_rounds[PARAM_K];
	cl_kernel k_sols;

	cl_mem buf_ht[2], buf_sols, buf_dbg;
	size_t global_ws;
	size_t local_work_size = 64;

	sols_t	*sols;

	bool init(cl_device_id dev, unsigned threadsNum, unsigned threadsPerBlock);
	
	~OclContext() {
		clReleaseMemObject(buf_dbg);
		clReleaseMemObject(buf_ht[0]);
		clReleaseMemObject(buf_ht[1]);
		free(sols);
	}
};

cl_mem check_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
	void *host_ptr);

bool OclContext::init(
	cl_device_id dev,
	unsigned int threadsNum,
	unsigned int threadsPerBlock)
{
	cl_int error;

	queue = clCreateCommandQueue(_context, dev, 0, &error);

#ifdef ENABLE_DEBUG
	size_t              dbg_size = NR_ROWS;
#else
	size_t              dbg_size = 1;
#endif

	buf_dbg = check_clCreateBuffer(_context, CL_MEM_READ_WRITE |
		CL_MEM_HOST_NO_ACCESS, dbg_size, NULL);
	buf_ht[0] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[1] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_sols = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, sizeof(sols_t),
		NULL);


	fprintf(stderr, "Hash tables will use %.1f MB\n", 2.0 * HT_SIZE / 1e6);

	k_init_ht = clCreateKernel(_program, "kernel_init_ht", &error);
	for (unsigned i = 0; i < WK; i++) {
		char kernelName[128];
		sprintf(kernelName, "kernel_round%d", i);
		k_rounds[i] = clCreateKernel(_program, kernelName, &error);
	}

	sols = (sols_t *)malloc(sizeof(*sols));

	k_sols = clCreateKernel(_program, "kernel_sols", &error);
	return true;
}

///
int             verbose = 0;
uint32_t	show_encoded = 0;

cl_mem check_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
	void *host_ptr)
{
	cl_int	status;
	cl_mem	ret;
	ret = clCreateBuffer(ctx, flags, size, host_ptr, &status);
	if (status != CL_SUCCESS || !ret)
		printf("clCreateBuffer (%d)\n", status);
	return ret;
}

void check_clSetKernelArg(cl_kernel k, cl_uint a_pos, cl_mem *a)
{
	cl_int	status;
	status = clSetKernelArg(k, a_pos, sizeof(*a), a);
	if (status != CL_SUCCESS)
		printf("clSetKernelArg (%d)\n", status);
}

void check_clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel k, cl_uint
	work_dim, const size_t *global_work_offset, const size_t
	*global_work_size, const size_t *local_work_size, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_uint	status;
	status = clEnqueueNDRangeKernel(queue, k, work_dim, global_work_offset,
		global_work_size, local_work_size, num_events_in_wait_list,
		event_wait_list, event);
	if (status != CL_SUCCESS)
		printf("clEnqueueNDRangeKernel (%d)\n", status);
}

void check_clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, cl_bool
	blocking_read, size_t offset, size_t size, void *ptr, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_int	status;
	status = clEnqueueReadBuffer(queue, buffer, blocking_read, offset,
		size, ptr, num_events_in_wait_list, event_wait_list, event);
	if (status != CL_SUCCESS)
		printf("clEnqueueReadBuffer (%d)\n", status);
}

void hexdump(uint8_t *a, uint32_t a_len)
{
	for (uint32_t i = 0; i < a_len; i++)
		fprintf(stderr, "%02x", a[i]);
}

char *s_hexdump(const void *_a, uint32_t a_len)
{
	const uint8_t	*a = (uint8_t	*)_a;
	static char		buf[1024];
	uint32_t		i;
	for (i = 0; i < a_len && i + 2 < sizeof(buf); i++)
		sprintf(buf + i * 2, "%02x", a[i]);
	buf[i * 2] = 0;
	return buf;
}

uint8_t hex2val(const char *base, size_t off)
{
	const char          c = base[off];
	if (c >= '0' && c <= '9')           return c - '0';
	else if (c >= 'a' && c <= 'f')      return 10 + c - 'a';
	else if (c >= 'A' && c <= 'F')      return 10 + c - 'A';
	printf("Invalid hex char at offset %zd: ...%c...\n", off, c);
	return 0;
}

unsigned nr_compute_units(const char *gpu)
{
	if (!strcmp(gpu, "rx480")) return 36;
	fprintf(stderr, "Unknown GPU: %s\n", gpu);
	return 0;
}

static void compress(uint8_t *out, uint32_t *inputs, uint32_t n)
{
	uint32_t byte_pos = 0;
	int32_t bits_left = PREFIX + 1;
	uint8_t x = 0;
	uint8_t x_bits_used = 0;
	uint8_t *pOut = out;
	while (byte_pos < n)
	{
		if (bits_left >= 8 - x_bits_used)
		{
			x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used);
			bits_left -= 8 - x_bits_used;
			x_bits_used = 8;
		}
		else if (bits_left > 0)
		{
			uint32_t mask = ~(-1 << (8 - x_bits_used));
			mask = ((~mask) >> bits_left) & mask;
			x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask;
			x_bits_used += bits_left;
			bits_left = 0;
		}
		else if (bits_left <= 0)
		{
			assert(!bits_left);
			byte_pos++;
			bits_left = PREFIX + 1;
		}
		if (x_bits_used == 8)
		{
			*pOut++ = x;
			x = x_bits_used = 0;
		}
	}
}

void get_program_build_log(cl_program program, cl_device_id device)
{
	cl_int		status;
	char	        val[2 * 1024 * 1024];
	size_t		ret = 0;
	status = clGetProgramBuildInfo(program, device,
		CL_PROGRAM_BUILD_LOG,
		sizeof(val),	// size_t param_value_size
		&val,		// void *param_value
		&ret);		// size_t *param_value_size_ret
	if (status != CL_SUCCESS)
		printf("clGetProgramBuildInfo (%d)\n", status);
	fprintf(stderr, "%s\n", val);
}

size_t select_work_size_blake(void)
{
	size_t              work_size =
		64 * /* thread per wavefront */
		BLAKE_WPS * /* wavefront per simd */
		4 * /* simd per compute unit */
		nr_compute_units("rx480");
	// Make the work group size a multiple of the nr of wavefronts, while
	// dividing the number of inputs. This results in the worksize being a
	// power of 2.
	while (NR_INPUTS % work_size)
		work_size += 64;
	//debug("Blake: work size %zd\n", work_size);
	return work_size;
}

static void init_ht(cl_command_queue queue, cl_kernel k_init_ht, cl_mem buf_ht)
{
	size_t      global_ws = NR_ROWS;
	size_t      local_ws = 64;
	cl_int      status;
#if 0
	uint32_t    pat = -1;
	status = clEnqueueFillBuffer(queue, buf_ht, &pat, sizeof(pat), 0,
		NR_ROWS * NR_SLOTS * SLOT_LEN,
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL);	// cl_event	*event
	if (status != CL_SUCCESS)
		fatal("clEnqueueFillBuffer (%d)\n", status);
#endif
	status = clSetKernelArg(k_init_ht, 0, sizeof(buf_ht), &buf_ht);
	if (status != CL_SUCCESS)
		printf("clSetKernelArg (%d)\n", status);
	check_clEnqueueNDRangeKernel(queue, k_init_ht,
		1,		// cl_uint	work_dim
		NULL,	// size_t	*global_work_offset
		&global_ws,	// size_t	*global_work_size
		&local_ws,	// size_t	*local_work_size
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL);	// cl_event	*event
}


/*
** Sort a pair of binary blobs (a, b) which are consecutive in memory and
** occupy a total of 2*len 32-bit words.
**
** a            points to the pair
** len          number of 32-bit words in each pair
*/
void sort_pair(uint32_t *a, uint32_t len)
{
	uint32_t    *b = a + len;
	uint32_t     tmp, need_sorting = 0;
	for (uint32_t i = 0; i < len; i++)
		if (need_sorting || a[i] > b[i])
		{
			need_sorting = 1;
			tmp = a[i];
			a[i] = b[i];
			b[i] = tmp;
		}
		else if (a[i] < b[i])
			return;
}
static uint32_t verify_sol(sols_t *sols, unsigned sol_i)
{
	uint32_t  *inputs = sols->values[sol_i];
	uint32_t  seen_len = (1 << (PREFIX + 1)) / 8;
	uint8_t seen[(1 << (PREFIX + 1)) / 8];
	uint32_t  i;
	uint8_t tmp;
	// look for duplicate inputs
	memset(seen, 0, seen_len);
	for (i = 0; i < (1 << PARAM_K); i++)
	{
		tmp = seen[inputs[i] / 8];
		seen[inputs[i] / 8] |= 1 << (inputs[i] & 7);
		if (tmp == seen[inputs[i] / 8])
		{
			// at least one input value is a duplicate
			sols->valid[sol_i] = 0;
			return 0;
		}
	}
	// the valid flag is already set by the GPU, but set it again because
	// I plan to change the GPU code to not set it
	sols->valid[sol_i] = 1;
	// sort the pairs in place
	for (uint32_t level = 0; level < PARAM_K; level++)
		for (i = 0; i < (1 << PARAM_K); i += (2 << level))
			sort_pair(&inputs[i], 1 << level);
	return 1;
}



ocl_silentarmy::ocl_silentarmy(int platf_id, int dev_id) {
	platform_id = platf_id;
	device_id = dev_id;
	// TODO 
	threadsNum = 8192;
	wokrsize = 128; // 256;
}

std::string ocl_silentarmy::getdevinfo() {
	/*TODO get name*/
	return "GPU_ID(" + std::to_string(device_id)+ ")";
}

// STATICS START
int ocl_silentarmy::getcount() { /*TODO*/
	return 0;
}

void ocl_silentarmy::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version) { /*TODO*/ }

void ocl_silentarmy::start(ocl_silentarmy& device_context) {
	/*TODO*/
	device_context.is_init_success = false;
	device_context.oclc = new OclContext();

	std::vector<cl_device_id> allGpus;
	if (!clInitialize(device_context.platform_id, allGpus)) {
		return;
	}

	// this is kinda stupid but it works
	std::vector<cl_device_id> gpus;
	for (unsigned i = 0; i < allGpus.size(); ++i) {
		if (i == device_context.device_id) {
			printf("Using device %d as GPU %d\n", i, (int)gpus.size());
			device_context.oclc->_dev_id = allGpus[i];
			gpus.push_back(allGpus[i]);
		}
	}

	if (!gpus.size()){
		printf("Device id %d not found\n", device_context.device_id);
		return;
	}

	// context create
	for (unsigned i = 0; i < gpus.size(); i++) {
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)device_context.oclc->platform_id, 0 };
		cl_int error;
		device_context.oclc->_context = clCreateContext(NULL, 1, &gpus[i], 0, 0, &error);
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
		sprintf(kernelName, "silentarmy_gpu%u.bin", (unsigned)i);
		if (!clCompileKernel(device_context.oclc->_context,
			gpus[i],
			kernelName,
			{ "zcash/gpu/kernel.cl" },
			"",
			&binstatus[i],
			&device_context.oclc->_program)) {
			return;
		}
	}

	for (unsigned i = 0; i < gpus.size(); ++i) {
		if (binstatus[i] == CL_SUCCESS) {
			if (!device_context.oclc->init(gpus[i], device_context.threadsNum, device_context.wokrsize)) {
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

void ocl_silentarmy::stop(ocl_silentarmy& device_context) {
	if (device_context.oclc != nullptr) delete device_context.oclc;
}

void ocl_silentarmy::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef,
	ocl_silentarmy& device_context) {

	unsigned char context[140];
	memset(context, 0, 140);
	memcpy(context, tequihash_header, tequihash_header_len);
	memcpy(context + tequihash_header_len, nonce, nonce_len);

	OclContext *miner = device_context.oclc;
	clFlush(miner->queue);

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)context, 128, 0);

	cl_mem buf_blake_st;
	buf_blake_st = check_clCreateBuffer(miner->_context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(blake2b_state_s), &initialCtx);


	for (unsigned round = 0; round < PARAM_K; round++)
	{
		if (round < 2)
			init_ht(miner->queue, miner->k_init_ht, miner->buf_ht[round % 2]);
		if (!round)
		{
			check_clSetKernelArg(miner->k_rounds[round], 0, &buf_blake_st);
			check_clSetKernelArg(miner->k_rounds[round], 1, &miner->buf_ht[round % 2]);
			miner->global_ws = select_work_size_blake();
		}
		else
		{
			check_clSetKernelArg(miner->k_rounds[round], 0, &miner->buf_ht[(round - 1) % 2]);
			check_clSetKernelArg(miner->k_rounds[round], 1, &miner->buf_ht[round % 2]);
			miner->global_ws = NR_ROWS;
		}
		check_clSetKernelArg(miner->k_rounds[round], 2, &miner->buf_dbg);
		if (round == PARAM_K - 1)
			check_clSetKernelArg(miner->k_rounds[round], 3, &miner->buf_sols);
		check_clEnqueueNDRangeKernel(miner->queue, miner->k_rounds[round], 1, NULL,
			&miner->global_ws, &miner->local_work_size, 0, NULL, NULL);
		// cancel function
		if (cancelf()) return;
	}
	check_clSetKernelArg(miner->k_sols, 0, &miner->buf_ht[0]);
	check_clSetKernelArg(miner->k_sols, 1, &miner->buf_ht[1]);
	check_clSetKernelArg(miner->k_sols, 2, &miner->buf_sols);
	miner->global_ws = NR_ROWS;
	check_clEnqueueNDRangeKernel(miner->queue, miner->k_sols, 1, NULL,
		&miner->global_ws, &miner->local_work_size, 0, NULL, NULL);

	check_clEnqueueReadBuffer(miner->queue, miner->buf_sols,
		CL_TRUE,	// cl_bool	blocking_read
		0,		// size_t	offset
		sizeof(*miner->sols),	// size_t	size
		miner->sols,	// void		*ptr
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL);	// cl_event	*event

	if (miner->sols->nr > MAX_SOLS)
		miner->sols->nr = MAX_SOLS;

	clReleaseMemObject(buf_blake_st);

	for (unsigned sol_i = 0; sol_i < miner->sols->nr; sol_i++) {
		verify_sol(miner->sols, sol_i);
	}

	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < miner->sols->nr; i++) {
		if (miner->sols->valid[i]) {
			compress(proof, (uint32_t *)(miner->sols->values[i]), 1 << PARAM_K);
			solutionf(std::vector<uint32_t>(0), 1344, proof);
		}
	}
	hashdonef();
}

// STATICS END

