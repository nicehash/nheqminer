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

	cl_platform_id gPlatform = 0;

	cl_command_queue queue;
	clBuffer<uint8_t> buf_ht0;
	clBuffer<uint8_t> buf_ht1;
	clBuffer<sols_t> buf_sols;
	clBuffer<debug_t> buf_dbg;

	cl_kernel k_init_ht;
	cl_kernel k_rounds[PARAM_K];
	cl_kernel k_sols;

	/*uint256 nonce;

	MinerInstance() {}*/
	bool init(cl_context context, cl_program program, cl_device_id dev, unsigned threadsNum, unsigned threadsPerBlock);
};

bool OclContext::init(cl_context context,
	cl_program program,
	cl_device_id dev,
	unsigned int threadsNum,
	unsigned int threadsPerBlock)
{
	cl_int error;

	_context = context;
	_program = program;
	queue = clCreateCommandQueue(context, dev, 0, &error);

#ifdef ENABLE_DEBUG
	size_t              dbg_size = NR_ROWS;
#else
	size_t              dbg_size = 1;
#endif  

	buf_dbg.init(context, dbg_size, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
	buf_ht0.init(context, HT_SIZE, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
	buf_ht1.init(context, HT_SIZE, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
	buf_sols.init(context, 1, CL_MEM_READ_WRITE);
	fprintf(stderr, "Hash tables will use %.1f MB\n", 2.0 * HT_SIZE / 1e6);

	k_init_ht = clCreateKernel(program, "kernel_init_ht", &error);
	for (unsigned i = 0; i < WK; i++) {
		char kernelName[128];
		sprintf(kernelName, "kernel_round%d", i);
		k_rounds[i] = clCreateKernel(program, kernelName, &error);
	}

	k_sols = clCreateKernel(program, "kernel_sols", &error);
	return true;
}


typedef struct  blake2b_state_s
{
	uint64_t    h[8];
	uint64_t    bytes;
}               blake2b_state_t;
void zcash_blake2b_init(blake2b_state_t *st, uint8_t hash_len,
	uint32_t n, uint32_t k);
void zcash_blake2b_update(blake2b_state_t *st, const uint8_t *_msg,
	uint32_t msg_len, uint32_t is_final);
void zcash_blake2b_final(blake2b_state_t *st, uint8_t *out, uint8_t outlen);

#include <stdint.h>
#include <string.h>
#include <assert.h>
//#include "blake.h"

static const uint32_t   blake2b_block_len = 128;
static const uint32_t   blake2b_rounds = 12;
static const uint64_t   blake2b_iv[8] =
{
	0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};
static const uint8_t    blake2b_sigma[12][16] =
{
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
};

/*
** Init the state according to Zcash parameters.
*/
void zcash_blake2b_init(blake2b_state_t *st, uint8_t hash_len,
	uint32_t n, uint32_t k)
{
	assert(n > k);
	assert(hash_len <= 64);
	st->h[0] = blake2b_iv[0] ^ (0x01010000 | hash_len);
	for (uint32_t i = 1; i <= 5; i++)
		st->h[i] = blake2b_iv[i];
	st->h[6] = blake2b_iv[6] ^ *(uint64_t *)"ZcashPoW";
	st->h[7] = blake2b_iv[7] ^ (((uint64_t)k << 32) | n);
	st->bytes = 0;
}

static uint64_t rotr64(uint64_t a, uint8_t bits)
{
	return (a >> bits) | (a << (64 - bits));
}

static void mix(uint64_t *va, uint64_t *vb, uint64_t *vc, uint64_t *vd,
	uint64_t x, uint64_t y)
{
	*va = (*va + *vb + x);
	*vd = rotr64(*vd ^ *va, 32);
	*vc = (*vc + *vd);
	*vb = rotr64(*vb ^ *vc, 24);
	*va = (*va + *vb + y);
	*vd = rotr64(*vd ^ *va, 16);
	*vc = (*vc + *vd);
	*vb = rotr64(*vb ^ *vc, 63);
}

/*
** Process either a full message block or the final partial block.
** Note that v[13] is not XOR'd because st->bytes is assumed to never overflow.
**
** _msg         pointer to message (must be zero-padded to 128 bytes if final block)
** msg_len      must be 128 (<= 128 allowed only for final partial block)
** is_final     indicate if this is the final block
*/
void zcash_blake2b_update(blake2b_state_t *st, const uint8_t *_msg,
	uint32_t msg_len, uint32_t is_final)
{
	const uint64_t      *m = (const uint64_t *)_msg;
	uint64_t            v[16];
	assert(msg_len <= 128);
	assert(st->bytes <= UINT64_MAX - msg_len);
	memcpy(v + 0, st->h, 8 * sizeof(*v));
	memcpy(v + 8, blake2b_iv, 8 * sizeof(*v));
	v[12] ^= (st->bytes += msg_len);
	v[14] ^= is_final ? -1 : 0;
	for (uint32_t round = 0; round < blake2b_rounds; round++)
	{
		const uint8_t   *s = blake2b_sigma[round];
		mix(v + 0, v + 4, v + 8, v + 12, m[s[0]], m[s[1]]);
		mix(v + 1, v + 5, v + 9, v + 13, m[s[2]], m[s[3]]);
		mix(v + 2, v + 6, v + 10, v + 14, m[s[4]], m[s[5]]);
		mix(v + 3, v + 7, v + 11, v + 15, m[s[6]], m[s[7]]);
		mix(v + 0, v + 5, v + 10, v + 15, m[s[8]], m[s[9]]);
		mix(v + 1, v + 6, v + 11, v + 12, m[s[10]], m[s[11]]);
		mix(v + 2, v + 7, v + 8, v + 13, m[s[12]], m[s[13]]);
		mix(v + 3, v + 4, v + 9, v + 14, m[s[14]], m[s[15]]);
	}
	for (uint32_t i = 0; i < 8; i++)
		st->h[i] ^= v[i] ^ v[i + 8];
}

void zcash_blake2b_final(blake2b_state_t *st, uint8_t *out, uint8_t outlen)
{
	assert(outlen <= 64);
	memcpy(out, st->h, outlen);
}

int             verbose = 0;
uint32_t	show_encoded = 0;
uint64_t	nr_nonces = 1;
uint32_t	do_list_gpu = 0;
uint32_t	gpu_to_use = 0;


uint64_t parse_num(char *str)
{
	char	*endptr;
	uint64_t	n;
	n = strtoul(str, &endptr, 0);
	if (endptr == str || *endptr)
		printf("'%s' is not a valid number\n", str);
	return n;
}

uint64_t now(void)
{
	/*struct timeval	tv;
	gettimeofday(&tv, NULL);
	return (uint64_t)tv.tv_sec * 1000 * 1000 + tv.tv_usec;*/
	return 0;
}

void show_time(uint64_t t0)
{
	uint64_t            t1;
	t1 = now();
	fprintf(stderr, "Elapsed time: %.1f msec\n", (t1 - t0) / 1e3);
}

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

void dump(const char *fname, void *data, size_t len)
{
	/*int			fd;
	ssize_t		ret;
	if (-1 == (fd = open(fname, O_WRONLY | O_CREAT | O_TRUNC, 0666)))
	printf("%s: %s\n", fname, strerror(errno));
	ret = write(fd, data, len);
	if (ret == -1)
	printf("write: %s: %s\n", fname, strerror(errno));
	if ((size_t)ret != len)
	printf("%s: partial write\n", fname);
	if (-1 == close(fd))
	printf("close: %s: %s\n", fname, strerror(errno));*/
}

void get_program_bins(cl_program program)
{
	cl_int		status;
	size_t		sizes;
	unsigned char	*p;
	size_t		ret = 0;
	status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
		sizeof(sizes),	// size_t param_value_size
		&sizes,		// void *param_value
		&ret);		// size_t *param_value_size_ret
	if (status != CL_SUCCESS)
		printf("clGetProgramInfo(sizes) (%d)\n", status);
	if (ret != sizeof(sizes))
		printf("clGetProgramInfo(sizes) did not fill sizes (%d)\n", status);
	printf("Program binary size is %zd bytes\n", sizes);
	p = (unsigned char *)malloc(sizes);
	status = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
		sizeof(p),	// size_t param_value_size
		&p,		// void *param_value
		&ret);	// size_t *param_value_size_ret
	if (status != CL_SUCCESS)
		printf("clGetProgramInfo (%d)\n", status);
	dump("dump.co", p, sizes);
	printf("program: %02x%02x%02x%02x...\n", p[0], p[1], p[2], p[3]);
}

void print_device_info(unsigned i, cl_device_id d)
{
	char	name[256];
	size_t	len = 0;
	int r;
	r = clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), &name,
		&len);
	if (r)
		printf("clGetDeviceInfo (%d)\n", r);
	printf("ID %d: %s\n", i, name);
}

//void examine_dbg(cl_command_queue queue, cl_mem buf_dbg, size_t dbg_size)
//{
//	debug_t     *dbg;
//	size_t      dropped_coll_total, dropped_stor_total;
//	if (verbose < 2)
//		return;
//	dbg = (debug_t *)malloc(dbg_size);
//	if (!dbg)
//		printf("malloc: %s\n", strerror(errno));
//	check_clEnqueueReadBuffer(queue, buf_dbg,
//		CL_TRUE,	// cl_bool	blocking_read
//		0,		// size_t	offset
//		dbg_size,   // size_t	size
//		dbg,	// void		*ptr
//		0,		// cl_uint	num_events_in_wait_list
//		NULL,	// cl_event	*event_wait_list
//		NULL);	// cl_event	*event
//	dropped_coll_total = dropped_stor_total = 0;
//	for (unsigned tid = 0; tid < dbg_size / sizeof(*dbg); tid++)
//	{
//		dropped_coll_total += dbg[tid].dropped_coll;
//		dropped_stor_total += dbg[tid].dropped_stor;
//		if (0 && (dbg[tid].dropped_coll || dbg[tid].dropped_stor))
//			printf("thread %6d: dropped_coll %zd dropped_stor %zd\n", tid,
//			dbg[tid].dropped_coll, dbg[tid].dropped_stor);
//	}
//	printf("Dropped: %zd (coll) %zd (stor)\n",
//		dropped_coll_total, dropped_stor_total);
//	free(dbg);
//}

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

static void init_ht(cl_command_queue queue, cl_kernel k_init_ht, clBuffer<uint8_t> &buf_ht)
{
	size_t global_ws = NR_ROWS;
	size_t local_ws = 64;
	cl_int status;
	OCL(clSetKernelArg(k_init_ht, 0, sizeof(cl_mem), &buf_ht.DeviceData));
	OCL(clEnqueueNDRangeKernel(queue, k_init_ht,
		1,    // cl_uint  work_dim
		NULL, // size_t *global_work_offset
		&global_ws, // size_t *global_work_size
		&local_ws,  // size_t *local_work_size
		0,    // cl_uint  num_events_in_wait_list
		NULL, // cl_event *event_wait_list
		NULL));  // cl_event *event
}

/*
** Print on stdout a hex representation of the encoded solution as per the
** zcash protocol specs (512 x 21-bit inputs).
**
** inputs       array of 32-bit inputs
** n            number of elements in array
*/
void print_encoded_sol(uint32_t *inputs, uint32_t n)
{
	uint32_t byte_pos = 0;
	int32_t bits_left = PREFIX + 1;
	uint8_t x = 0;
	uint8_t x_bits_used = 0;
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
			//assert(!bits_left);
			byte_pos++;
			bits_left = PREFIX + 1;
		}
		if (x_bits_used == 8)
		{
			printf("%02x", x);
			x = x_bits_used = 0;
		}
	}
	printf("\n");
	fflush(stdout);
}

void print_sol(uint32_t *values, uint64_t *nonce)
{
	uint32_t	show_n_sols;
	show_n_sols = (1 << PARAM_K);
	if (verbose < 2)
		show_n_sols = MIN(10, show_n_sols);
	fprintf(stderr, "Soln:");
	// for brievity, only print "small" nonces
	if (*nonce < (1UL << 32))
		fprintf(stderr, " 0x%lx:", *nonce);
	for (unsigned i = 0; i < show_n_sols; i++)
		fprintf(stderr, " %x", values[i]);
	fprintf(stderr, "%s\n", (show_n_sols != (1 << PARAM_K) ? "..." : ""));
}

int sol_cmp(const void *_a, const void *_b)
{
	const uint32_t	*a = (uint32_t	*)_a;
	const uint32_t	*b = (uint32_t	*)_b;
	for (uint32_t i = 0; i < (1 << PARAM_K); i++)
	{
		if (*a != *b)
			return *a - *b;
		a++;
		b++;
	}
	return 0;
}

/*
** Print all solutions.
*/
void print_sols(sols_t *all_sols, uint64_t *nonce, uint32_t nr_valid_sols)
{
	uint8_t		*valid_sols;
	uint32_t		counted;
	valid_sols = (uint8_t *)malloc(nr_valid_sols * SOL_SIZE);
	if (!valid_sols)
		printf("malloc: %s\n", strerror(errno));
	counted = 0;
	for (uint32_t i = 0; i < all_sols->nr; i++)
		if (all_sols->valid[i])
		{
			if (counted >= nr_valid_sols)
				printf("Bug: more than %d solutions\n", nr_valid_sols);
			memcpy(valid_sols + counted * SOL_SIZE, all_sols->values[i],
				SOL_SIZE);
			counted++;
		}
	//assert(counted == nr_valid_sols);
	// sort the solutions amongst each other, to make silentarmy's output
	// deterministic and testable
	qsort(valid_sols, nr_valid_sols, SOL_SIZE, sol_cmp);
	for (uint32_t i = 0; i < nr_valid_sols; i++)
	{
		uint32_t	*inputs = (uint32_t *)(valid_sols + i * SOL_SIZE);
		if (show_encoded)
			print_encoded_sol(inputs, 1 << PARAM_K);
		if (verbose)
			print_sol(inputs, nonce);
	}
	free(valid_sols);
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

void list_gpu(cl_device_id *devices, cl_uint nr)
{
	(void)devices;
	printf("Found %d GPU device%s\n", nr, (nr != 1) ? "s" : "");
	for (uint32_t i = 0; i < nr; i++)
		print_device_info(i, devices[i]);
}

ocl_silentarmy::ocl_silentarmy(int platf_id, int dev_id) { /*TODO*/
	platform_id = platf_id;
	device_id = dev_id;
	// TODO 
	threadsNum = 8192;
	wokrsize = 128; // 256;
	//threadsperblock = 128;
}

std::string ocl_silentarmy::getdevinfo() { /*TODO*/
	return "TODO";
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

	cl_context gContext[64] = { 0 };
	cl_program gProgram[64] = { 0 };


	std::vector<cl_device_id> allGpus;
	// use only AMD platforms
	const char *platformName = "AMD Accelerated Parallel Processing";
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
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)device_context.oclc->gPlatform, 0 };
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
		sprintf(kernelName, "silentarmy_gpu%u.bin", (unsigned)i);
		if (!clCompileKernel(gContext[i],
			gpus[i],
			kernelName,
			{ "zcash/gpu/kernel.cl" },
			"",
			&binstatus[i],
			&gProgram[i])) {
			return;
		}
	}

	for (unsigned i = 0; i < gpus.size(); ++i) {
		if (binstatus[i] == CL_SUCCESS) {
			if (!device_context.oclc->init(gContext[i], gProgram[i], gpus[i], device_context.threadsNum, device_context.wokrsize)) {
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

void ocl_silentarmy::stop(ocl_silentarmy& device_context) { /*TODO*/


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

	/*unsigned char context[140];
	memset(context, 0, 140);
	memcpy(context, tequihash_header, tequihash_header_len);
	memcpy(context + tequihash_header_len, nonce, nonce_len);*/

	OclContext *miner = device_context.oclc;
	clFlush(miner->queue);

	//C++ START
	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)&tequihash_header, 128, 0);

	//miner->nonce = header.nNonce;
	size_t global_ws;
	size_t local_work_size = 64;
	for (unsigned round = 0; round < PARAM_K; round++) {
		clBuffer<uint8_t> &bufHtFirst = (round % 2 == 0) ? miner->buf_ht0 : miner->buf_ht1;
		clBuffer<uint8_t> &bufHtSecond = (round % 2 == 0) ? miner->buf_ht1 : miner->buf_ht0;

		init_ht(miner->queue, miner->k_init_ht, bufHtFirst);
		if (round == 0) {
			OCL(clSetKernelArg(miner->k_rounds[round], 0, sizeof(cl_mem), &bufHtFirst.DeviceData));
			OCL(clSetKernelArg(miner->k_rounds[round], 1, sizeof(cl_mem), &bufHtFirst.DeviceData));
			OCL(clSetKernelArg(miner->k_rounds[round], 3, sizeof(cl_ulong), &initialCtx.h[0]));
			OCL(clSetKernelArg(miner->k_rounds[round], 4, sizeof(cl_ulong), &initialCtx.h[1]));
			OCL(clSetKernelArg(miner->k_rounds[round], 5, sizeof(cl_ulong), &initialCtx.h[2]));
			OCL(clSetKernelArg(miner->k_rounds[round], 6, sizeof(cl_ulong), &initialCtx.h[3]));
			OCL(clSetKernelArg(miner->k_rounds[round], 7, sizeof(cl_ulong), &initialCtx.h[4]));
			OCL(clSetKernelArg(miner->k_rounds[round], 8, sizeof(cl_ulong), &initialCtx.h[5]));
			OCL(clSetKernelArg(miner->k_rounds[round], 9, sizeof(cl_ulong), &initialCtx.h[6]));
			OCL(clSetKernelArg(miner->k_rounds[round], 10, sizeof(cl_ulong), &initialCtx.h[7]));
			global_ws = select_work_size_blake();
		}
		else {
			OCL(clSetKernelArg(miner->k_rounds[round], 0, sizeof(cl_mem), &bufHtSecond.DeviceData));
			OCL(clSetKernelArg(miner->k_rounds[round], 1, sizeof(cl_mem), &bufHtFirst.DeviceData));
			global_ws = NR_ROWS;
		}

		OCL(clSetKernelArg(miner->k_rounds[round], 2, sizeof(cl_mem), &miner->buf_dbg.DeviceData));
		OCL(clEnqueueNDRangeKernel(miner->queue, miner->k_rounds[round], 1, NULL, &global_ws, &local_work_size, 0, NULL, NULL));
	}

	OCL(clSetKernelArg(miner->k_sols, 0, sizeof(cl_mem), &miner->buf_ht0.DeviceData));
	OCL(clSetKernelArg(miner->k_sols, 1, sizeof(cl_mem), &miner->buf_ht1.DeviceData));
	OCL(clSetKernelArg(miner->k_sols, 2, sizeof(cl_mem), &miner->buf_sols.DeviceData));
	global_ws = NR_ROWS;
	OCL(clEnqueueNDRangeKernel(miner->queue, miner->k_sols, 1, NULL, &global_ws, &local_work_size, 0, NULL, NULL));
	//C++ END

	miner->buf_sols.copyToHost(miner->queue, true);
	sols_t *sols = miner->buf_sols.HostData;
	if (sols->nr > MAX_SOLS)
		sols->nr = MAX_SOLS;

	for (unsigned sol_i = 0; sol_i < sols->nr; sol_i++)
		verify_sol(sols, sol_i);

	// TODO send compressed or non compressed data
	uint32_t nsols = 0;
	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < sols->nr; i++) {
		if (sols->valid[i]) {
			std::vector<uint32_t> index_vector(PROOFSIZE);
			for (uint32_t el = 0; el < PROOFSIZE; el++) {
				index_vector[i] = sols->values[i][el];
			}

			solutionf(index_vector, DIGITBITS, nullptr);
			if (cancelf()) return;
			//compress(proof, (uint32_t *)(sols->values[i]), 1 << PARAM_K);
			// TODO remove
			nsols++;
		}
	}
	// TODO remove
	printf("solution num %d\n", nsols);
}

void ocl_silentarmy::print_opencl_devices() {
	/*TODO*/
}
// STATICS END

