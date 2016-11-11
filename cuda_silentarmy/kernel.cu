#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <functional>
#include <vector>
#include <iostream>
#include <stdint.h>

#include "sa_cuda_context.hpp"
#include "param.h"
#include "sa_blake.h"

#define WN PARAM_N
#define WK PARAM_K

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))

#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long long  ulong;
typedef unsigned short ushort;
typedef uint32_t u32;

typedef struct sols_s
{
	uint nr;
	uint likely_invalids;
	uchar valid[MAX_SOLS];
	uint values[MAX_SOLS][(1 << PARAM_K)];
} sols_t;

__constant__ ulong blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};


__global__
void kernel_init_ht(char *ht)
{
	static uint stride = NR_SLOTS * SLOT_LEN;//(((1 << (((200 / (9 + 1)) + 1) - 20)) * 6) * 32);
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = tid; i < NR_ROWS; i += (gridDim.x * blockDim.x)) {
		*(uint*)(ht + i * stride) = 0;
	}
}

__device__ uint ht_store(uint round, char *ht, uint i, ulong xi0, ulong xi1, ulong xi2, ulong xi3)
{
	uint		row;
	char       *p;
	uint       cnt;
	if (!(round & 1))
		row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
	else
		row = ((xi0 & 0xf0000) >> 0) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
	xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
	p = ht + row * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 6) * 32;
	cnt = atomicAdd((uint*)p, 1);
	if (cnt >= ((1 << (((200 / (9 + 1)) + 1) - 20)) * 6))
		return 1;
	p += cnt * 32 + (8 + ((round) / 2) * 4);
	// store "i" (always 4 bytes before Xi)
	*(uint *)(p - 4) = i;
	if (round == 0 || round == 1) {
		// store 24 bytes
		*(ulong *)(p + 0) = xi0;
		*(ulong *)(p + 8) = xi1;
		*(ulong *)(p + 16) = xi2;
	} else if (round == 2) {
		// store 20 bytes
		*(uint *)(p + 0) = xi0;
		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
	} else if (round == 3) {
		// store 16 bytes
		*(uint *)(p + 0) = xi0;
		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(uint *)(p + 12) = (xi1 >> 32);
	} else if (round == 4) {
		// store 16 bytes
		*(ulong *)(p + 0) = xi0;
		*(ulong *)(p + 8) = xi1;
	} else if (round == 5) {
		// store 12 bytes
		*(ulong *)(p + 0) = xi0;
		*(uint *)(p + 8) = xi1;
	} else if (round == 6 || round == 7) {
		// store 8 bytes
		*(uint *)(p + 0) = xi0;
		*(uint *)(p + 4) = (xi0 >> 32);
	} else if (round == 8) {
		// store 4 bytes
		*(uint *)(p + 0) = xi0;
	}
	return 0;
}


#define rotate(a, bits) ((a) << (bits)) | ((a) >> (64 - (bits)))

#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
    vd = rotate((vd ^ va), 64 - 32); \
    vc = (vc + vd); \
    vb = rotate((vb ^ vc), 64 - 24); \
    va = (va + vb + y); \
    vd = rotate((vd ^ va), 64 - 16); \
    vc = (vc + vd); \
    vb = rotate((vb ^ vc), 64 - 63);

__global__ void 
kernel_round0(ulong *blake_state, char *ht, uint *debug)
{
	uint                tid = blockIdx.x*blockDim.x + threadIdx.x;
	ulong               v[16];
	uint                inputs_per_thread = NR_INPUTS / (gridDim.x * blockDim.x);
	uint                input = tid * inputs_per_thread;
	uint                input_end = (tid + 1) * inputs_per_thread;
	uint                dropped = 0;
	while (input < input_end) {
		// shift "i" to occupy the high 32 bits of the second ulong word in the
		// message block
		ulong word1 = (ulong)input << 32;
		// init vector v
		v[0] = blake_state[0];
		v[1] = blake_state[1];
		v[2] = blake_state[2];
		v[3] = blake_state[3];
		v[4] = blake_state[4];
		v[5] = blake_state[5];
		v[6] = blake_state[6];
		v[7] = blake_state[7];
		v[8] = blake_iv[0];
		v[9] = blake_iv[1];
		v[10] = blake_iv[2];
		v[11] = blake_iv[3];
		v[12] = blake_iv[4];
		v[13] = blake_iv[5];
		v[14] = blake_iv[6];
		v[15] = blake_iv[7];
		// mix in length of data
		v[12] ^= 140 + 4 /* length of "i" */;
		// last block
		v[14] ^= (ulong)-1;

		// round 1
		mix(v[0], v[4], v[8], v[12], 0, word1);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 2
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], word1, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 3
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, word1);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 4
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, word1);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 5
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, word1);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 6
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], word1, 0);
		// round 7
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], word1, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 8
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, word1);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 9
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], word1, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 10
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], word1, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 11
		mix(v[0], v[4], v[8], v[12], 0, word1);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 12
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], word1, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);

		// compress v into the blake state; this produces the 50-byte hash
		// (two Xi values)
		ulong h[7];
		h[0] = blake_state[0] ^ v[0] ^ v[8];
		h[1] = blake_state[1] ^ v[1] ^ v[9];
		h[2] = blake_state[2] ^ v[2] ^ v[10];
		h[3] = blake_state[3] ^ v[3] ^ v[11];
		h[4] = blake_state[4] ^ v[4] ^ v[12];
		h[5] = blake_state[5] ^ v[5] ^ v[13];
		h[6] = (blake_state[6] ^ v[6] ^ v[14]) & 0xffff;

		// store the two Xi values in the hash table
		dropped += ht_store(0, ht, input * 2,
			h[0],
			h[1],
			h[2],
			h[3]);
		dropped += ht_store(0, ht, input * 2 + 1,
			(h[3] >> 8) | (h[4] << (64 - 8)),
			(h[4] >> 8) | (h[5] << (64 - 8)),
			(h[5] >> 8) | (h[6] << (64 - 8)),
			(h[6] >> 8));

		input++;
	  }
}

__device__ ulong half_aligned_long(ulong *p, uint offset)
{
	return
		(((ulong)*(uint *)((char *)p + offset + 0)) << 0) |
		(((ulong)*(uint *)((char *)p + offset + 4)) << 32);
}

/*
** Access a well-aligned int.
*/
__device__ uint well_aligned_int(ulong *_p, uint offset)
{
	char *p = (char *)_p;
	return *(uint *)(p + offset);
}

/*
** XOR a pair of Xi values computed at "round - 1" and store the result in the
** hash table being built for "round". Note that when building the table for
** even rounds we need to skip 1 padding byte present in the "round - 1" table
** (the "0xAB" byte mentioned in the description at the top of this file.) But
** also note we can't load data directly past this byte because this would
** cause an unaligned memory access which is undefined per the OpenCL spec.
**
** Return 0 if successfully stored, or 1 if the row overflowed.
*/
__device__ uint xor_and_store(uint round, char *ht_dst, uint row, uint slot_a, uint slot_b, ulong *a, ulong *b)
{
	ulong	xi0, xi1, xi2;
	// Note: for NR_ROWS_LOG == 20, for odd rounds, we could optimize by not
	// storing the byte containing bits from the previous PREFIX block for
	if (round == 1 || round == 2)
	{
		// xor 24 bytes
		xi0 = *(a++) ^ *(b++);
		xi1 = *(a++) ^ *(b++);
		xi2 = *a ^ *b;
		if (round == 2)
		{
			// skip padding byte
			xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
			xi1 = (xi1 >> 8) | (xi2 << (64 - 8));
			xi2 = (xi2 >> 8);
		}
	}
	else if (round == 3)
	{
		// xor 20 bytes
		xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
		xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
		xi2 = well_aligned_int(a, 16) ^ well_aligned_int(b, 16);
	}
	else if (round == 4 || round == 5)
	{
		// xor 16 bytes
		xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
		xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
		xi2 = 0;
		if (round == 4)
		{
			// skip padding byte
			xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
			xi1 = (xi1 >> 8);
		}
	}
	else if (round == 6)
	{
		// xor 12 bytes
		xi0 = *a++ ^ *b++;
		xi1 = *(uint *)a ^ *(uint *)b;
		xi2 = 0;
		if (round == 6)
		{
			// skip padding byte
			xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
			xi1 = (xi1 >> 8);
		}
	}
	else if (round == 7 || round == 8)
	{
		// xor 8 bytes
		xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
		xi1 = 0;
		xi2 = 0;
		if (round == 8)
		{
			// skip padding byte
			xi0 = (xi0 >> 8);
		}
	}
	// invalid solutions (which start happenning in round 5) have duplicate
	// inputs and xor to zero, so discard them
	if (!xi0 && !xi1)
		return 0;
	return ht_store(round, ht_dst, ((row << 12) | ((slot_b & 0x3f) << 6) | (slot_a & 0x3f)),
		xi0, xi1, xi2, 0);
}

/*
** Execute one Equihash round. Read from ht_src, XOR colliding pairs of Xi,
** store them in ht_dst.
*/
__device__ void equihash_round(uint round, char *ht_src, char *ht_dst, uint *debug)
{
	uint                tid = blockIdx.x * blockDim.x + threadIdx.x;
	char				*p;
	uint                cnt;
	uint                i, j;
	uint				dropped_stor = 0;
	ulong				*a, *b;
	uint				xi_offset;
	static uint			size = NR_ROWS;
	static uint			stride = NR_SLOTS * SLOT_LEN;
	xi_offset = (8 + ((round - 1) / 2) * 4);

	for (uint ii = tid; ii < size; ii += (blockDim.x * gridDim.x)) {
		p = (ht_src + ii * stride);
		cnt = *(uint *)p;
		cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
		if (!cnt) {// no elements in row, no collisions
			continue;
		}
		// find collisions
		for (i = 0; i < cnt; i++) {
			for (j = i + 1; j < cnt; j++)
			{
				a = (ulong *)
					(ht_src + ii * stride + i * 32 + xi_offset);
				b = (ulong *)
					(ht_src + ii * stride + j * 32 + xi_offset);
				dropped_stor += xor_and_store(round, ht_dst, ii, i, j, a, b);
			}
		}
		if (round < 8) {
			// reset the counter in preparation of the next round
			*(uint *)(ht_src + ii * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 6) * 32) = 0;
		}
	}
}

__global__ void 
kernel_round1(char *ht_src, char *ht_dst, uint *debug)
{
	equihash_round(1, ht_src, ht_dst, debug);
}

__global__ void 
kernel_round2(char *ht_src, char *ht_dst, uint *debug)
{
	equihash_round(2, ht_src, ht_dst, debug);
}
__global__ void
kernel_round3(char *ht_src, char *ht_dst, uint *debug)
{
	equihash_round(3, ht_src, ht_dst, debug);
}

__global__ void
kernel_round4(char *ht_src, char *ht_dst, uint *debug)
{
	equihash_round(4, ht_src, ht_dst, debug);
}
__global__ void
kernel_round5(char *ht_src, char *ht_dst, uint *debug)
{
	equihash_round(5, ht_src, ht_dst, debug);
}
__global__ void
kernel_round6(char *ht_src, char *ht_dst, uint *debug)
{
	equihash_round(6, ht_src, ht_dst, debug);
}
__global__ void
kernel_round7(char *ht_src, char *ht_dst, uint *debug)
{
	equihash_round(7, ht_src, ht_dst, debug);
}

// kernel_round8 takes an extra argument, "sols"
__global__ void
kernel_round8(char *ht_src, char *ht_dst, uint *debug, sols_t *sols)
{
	uint                tid = blockIdx.x * blockDim.x + threadIdx.x;
	equihash_round(8, ht_src, ht_dst, debug);
	if (!tid)
		sols->nr = sols->likely_invalids = 0;
}

__device__ uint expand_ref(char *ht, uint xi_offset, uint row, uint slot)
{
	return *(uint *)(ht + row * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 6) * 32 + slot * 32 + xi_offset - 4);
}

__device__ uint expand_refs(uint *ins, uint nr_inputs, char **htabs, uint round)
{
	char *ht = htabs[round & 1];
	uint i = nr_inputs - 1;
	uint j = nr_inputs * 2 - 1;
	uint xi_offset = (8 + ((round) / 2) * 4);
	int dup_to_watch = -1;
	do
	{
		ins[j] = expand_ref(ht, xi_offset,
			(ins[i] >> 12), ((ins[i] >> 6) & 0x3f));
		ins[j - 1] = expand_ref(ht, xi_offset,
			(ins[i] >> 12), (ins[i] & 0x3f));
		if (!round) {
			if (dup_to_watch == -1) {
				dup_to_watch = ins[j];
			} else if (ins[j] == dup_to_watch || ins[j - 1] == dup_to_watch) {
				return 0;
			}
		}
		if (!i)
			break;
		i--;
		j -= 2;
	} while (1);
	return 1;
}

/*
** Verify if a potential solution is in fact valid.
*/
__device__ void potential_sol(char **htabs, sols_t *sols, uint ref0, uint ref1)
{
	uint	nr_values;
	uint	values_tmp[(1 << PARAM_K)];
	uint	sol_i;
	uint	i;
	nr_values = 0;
	values_tmp[nr_values++] = ref0;
	values_tmp[nr_values++] = ref1;
	uint round = PARAM_K - 1;
	do
	{
		round--;
		if (!expand_refs(values_tmp, nr_values, htabs, round)) {
			return;
		}
		nr_values *= 2;
	} while (round > 0);
	//solution looks valid
	sol_i = atomicAdd(&sols->nr, 1);
	if (sol_i >= MAX_SOLS) {
		return;
	}
	for (i = 0; i < (1 << PARAM_K); i++) {
		sols->values[sol_i][i] = values_tmp[i];
	}
	sols->valid[sol_i] = 1;
}

/*
** Scan the hash tables to find Equihash solutions.
*/
__global__
void kernel_sols(char *ht0, char *ht1, sols_t *sols)
{
	uint		tid = blockIdx.x * blockDim.x + threadIdx.x;
	char		*htabs[2] = { ht0, ht1 };
	uint		ht_i = (PARAM_K - 1) & 1; // table filled at last round
	uint		cnt;
	uint		xi_offset = xi_offset_for_round(PARAM_K - 1);
	uint		i, j;
	char		*a, *b;
	uint		ref_i, ref_j;
	// it's ok for the collisions array to be so small, as if it fills up
	// the potential solutions are likely invalid (many duplicate inputs)
	uint		coll;
	// in the final hash table, we are looking for a match on both the bits
	// part of the previous PREFIX colliding bits, and the last PREFIX bits.
	uint		mask = 0xffffff;
	uint ran = 0;
	for (uint ii = tid; ii < (uint)NR_ROWS; ii += (blockDim.x * gridDim.x)) {
		ran++;
		ulong		collisions[1];
		a = htabs[ht_i] + ii * NR_SLOTS * SLOT_LEN;
		cnt = *(uint *)a;
		cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
		coll = 0;
		a += xi_offset;
		for (i = 0; i < cnt; i++, a += 32) {
			for (j = i + 1, b = a + 32; j < cnt; j++, b += 32) {
				if (((*(uint *)a) & mask) == ((*(uint *)b) & mask)) {
					ref_i = *(uint *)(a - 4);
					ref_j = *(uint *)(b - 4);
					if (coll < sizeof(collisions) / sizeof(*collisions)) {
						collisions[coll++] = ((ulong)ref_i << 32) | ref_j;
					} else {
						atomicAdd(&sols->likely_invalids, 1);
					}
				}
			}
		}
		if (!coll) {
			continue;
		}
		for (i = 0; i < coll; i++) {
			potential_sol(htabs, sols, collisions[i] >> 32, collisions[i] & 0xffffffff);
		}
	}
}

struct __align__(64) c_context {
	char* buf_ht[2], *buf_sols, *buf_dbg;
	sols_t	*sols;
	u32 nthreads;
	size_t global_ws;


	c_context(const u32 n_threads) {
		nthreads = n_threads;
	}
	void* operator new(size_t i) {
		return _mm_malloc(i, 64);
	}
	void operator delete(void* p) {
		_mm_free(p);
	}
};

static size_t select_work_size_blake(void)
{
	size_t              work_size =
		64 * /* thread per wavefront */
		BLAKE_WPS * /* wavefront per simd */
		4 * /* simd per compute unit */
		36;
	// Make the work group size a multiple of the nr of wavefronts, while
	// dividing the number of inputs. This results in the worksize being a
	// power of 2.
	while (NR_INPUTS % work_size)
		work_size += 64;
	//debug("Blake: work size %zd\n", work_size);
	return work_size;
}

static void sort_pair(uint32_t *a, uint32_t len)
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

sa_cuda_context::sa_cuda_context(int tpb, int blocks, int id)
	: threadsperblock(tpb), totalblocks(blocks), device_id(id)
{
	checkCudaErrors(cudaSetDevice(device_id));
	checkCudaErrors(cudaDeviceReset());
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	eq = new c_context(threadsperblock * totalblocks);
#ifdef ENABLE_DEBUG
	size_t              dbg_size = NR_ROWS;
#else
	size_t              dbg_size = 1;
#endif

	checkCudaErrors(cudaMalloc((void**)&eq->buf_dbg, dbg_size));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_ht[0], HT_SIZE));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_ht[1], HT_SIZE));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_sols, sizeof(sols_t)));

	eq->sols = (sols_t *)malloc(sizeof(*eq->sols));
}

sa_cuda_context::~sa_cuda_context()
{
	checkCudaErrors(cudaSetDevice(device_id));
	checkCudaErrors(cudaDeviceReset());
	delete eq;
}

void sa_cuda_context::solve(const char * tequihash_header, unsigned int tequihash_header_len, const char * nonce, unsigned int nonce_len, std::function<bool()> cancelf, std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf, std::function<void(void)> hashdonef)
{
	checkCudaErrors(cudaSetDevice(device_id));

	unsigned char context[140];
	memset(context, 0, 140);
	memcpy(context, tequihash_header, tequihash_header_len);
	memcpy(context + tequihash_header_len, nonce, nonce_len);

	c_context *miner = eq;
	
	//FUNCTION<<<totalblocks, threadsperblock>>>(ARGUMENTS)

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)context, 128, 0);

	void* buf_blake_st;
	checkCudaErrors(cudaMalloc((void**)&buf_blake_st, sizeof(blake2b_state_s)));
	checkCudaErrors(cudaMemcpy(buf_blake_st, &initialCtx, sizeof(blake2b_state_s), cudaMemcpyHostToDevice));
	
	for (unsigned round = 0; round < PARAM_K; round++) {
		if (round < 2) {
			kernel_init_ht<<<totalblocks, threadsperblock>>>(miner->buf_ht[round & 1]);
		}
		if (!round)	{
			miner->global_ws = select_work_size_blake();
		} else {
			miner->global_ws = NR_ROWS;
		}
		// cancel function
		switch (round) {
		case 0:
			kernel_round0<<<totalblocks, threadsperblock>>>((ulong*)buf_blake_st, miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 1:
			kernel_round1<<<totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 2:
			kernel_round2<<< totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 3:
			kernel_round3<<<totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 4:
			kernel_round4<<<totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 5:
			kernel_round5<<<totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 6:
			kernel_round6<<<totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 7:
			kernel_round7<<<totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
			break;
		case 8:
			kernel_round8<<<totalblocks, threadsperblock>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg, (sols_t*)miner->buf_sols);
			break;
		}
		if (cancelf()) return;
	}
	kernel_sols<<<totalblocks, threadsperblock>>>(miner->buf_ht[0], miner->buf_ht[1], (sols_t*)miner->buf_sols);

	checkCudaErrors(cudaMemcpy(miner->sols, miner->buf_sols, sizeof(*miner->sols), cudaMemcpyDeviceToHost));

	if (miner->sols->nr > MAX_SOLS)
		miner->sols->nr = MAX_SOLS;

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