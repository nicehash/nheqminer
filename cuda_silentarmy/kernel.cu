#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <functional>
#include <vector>
#include <iostream>
#include <stdint.h>
#ifndef _MSC_VER
#include <mm_malloc.h>
#endif
#include "sa_cuda_context.hpp"
#include "param.h"
#include "sa_blake.h"



#define THRD 64
#define BLK (NR_ROWS/THRD)


#define WN PARAM_N
#define WK PARAM_K

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))

#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))

typedef uint32_t uint;
typedef uint8_t uchar;
typedef uint64_t ulong;
typedef uint16_t ushort;
typedef uint32_t u32;

//orig defines

#if NR_ROWS_LOG <= 16 && NR_SLOTS <= (1 << 8)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 16) | ((slot1 & 0xff) << 8) | (slot0 & 0xff))
#define DECODE_ROW(REF)   (REF >> 16)
#define DECODE_SLOT1(REF) ((REF >> 8) & 0xff)
#define DECODE_SLOT0(REF) (REF & 0xff)

#elif NR_ROWS_LOG == 18 && NR_SLOTS <= (1 << 7)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 14) | ((slot1 & 0x7f) << 7) | (slot0 & 0x7f))
#define DECODE_ROW(REF)   (REF >> 14)
#define DECODE_SLOT1(REF) ((REF >> 7) & 0x7f)
#define DECODE_SLOT0(REF) (REF & 0x7f)

#elif NR_ROWS_LOG == 19 && NR_SLOTS <= (1 << 6)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 13) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f)) /* 1 spare bit */
#define DECODE_ROW(REF)   (REF >> 13)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)

#elif NR_ROWS_LOG == 20 && NR_SLOTS <= (1 << 6)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 12) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f))
#define DECODE_ROW(REF)   (REF >> 12)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)

#else
#error "unsupported NR_ROWS_LOG"
#endif




__constant__ ulong blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

//OPENCL TO CUDA

#define __global
#define __local __shared__
#define get_global_id(F) (blockIdx.x * blockDim.x + threadIdx.x)
#define get_global_size(F) (gridDim.x * blockDim.x)
#define get_local_id(F) (threadIdx.x)
#define get_local_size(F) (blockDim.x)
#define barrier(F) __syncthreads()

//#define barrier(F) __threadfence()

#define atomic_add(A,X) atomicAdd(A,X)
#define atomic_inc(A) atomic_add(A,1)
#define atomic_sub(A,X) atomicSub(A,X)

__global__
void kernel_init_ht(__global char *ht, __global uint *rowCounters)
{
    rowCounters[get_global_id(0)] = 0;
}


__device__ uint ht_store(uint round, char *ht, uint i, ulong xi0, ulong xi1, ulong xi2, ulong xi3,uint *rowCounters)
{
	    uint    row;
    __global char       *p;
    uint                cnt;
#if NR_ROWS_LOG == 16
    if (!(round % 2))
	row = (xi0 & 0xffff);
    else
	// if we have in hex: "ab cd ef..." (little endian xi0) then this
	// formula computes the row as 0xdebc. it skips the 'a' nibble as it
	// is part of the PREFIX. The Xi will be stored starting with "ef...";
	// 'e' will be considered padding and 'f' is part of the current PREFIX
	row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
    else
	row = ((xi0 & 0xc0000) >> 2) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
    else
	row = ((xi0 & 0xe0000) >> 1) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
    else
	row = ((xi0 & 0xf0000) >> 0) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
    xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
    xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
    xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
    p = ht + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
     if (cnt >= NR_SLOTS)
       {
 	// avoid overflows
 	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
  	return 1;
       }

    p += cnt * SLOT_LEN + xi_offset_for_round(round);
    // store "i" (always 4 bytes before Xi)
    *(__global uint *)(p - 4) = i;
    if (round == 0 || round == 1)
      {
	// store 24 bytes
	*(__global ulong *)(p + 0) = xi0;
	*(__global ulong *)(p + 8) = xi1;
	*(__global ulong *)(p + 16) = xi2;
      }
    else if (round == 2)
      {
	// store 20 bytes
	*(__global uint *)(p + 0) = xi0;
	*(__global ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*(__global ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
      }
    else if (round == 3)
      {
	// store 16 bytes
	*(__global uint *)(p + 0) = xi0;
	*(__global ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*(__global uint *)(p + 12) = (xi1 >> 32);
      }
    else if (round == 4)
      {
	// store 16 bytes
	*(__global ulong *)(p + 0) = xi0;
	*(__global ulong *)(p + 8) = xi1;
      }
    else if (round == 5)
      {
	// store 12 bytes
	*(__global ulong *)(p + 0) = xi0;
	*(__global uint *)(p + 8) = xi1;
      }
    else if (round == 6 || round == 7)
      {
	// store 8 bytes
	*(__global uint *)(p + 0) = xi0;
	*(__global uint *)(p + 4) = (xi0 >> 32);
      }
    else if (round == 8)
      {
	// store 4 bytes
	*(__global uint *)(p + 0) = xi0;
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

__global__ 
void kernel_round0(__global ulong *blake_state, __global char *ht,
	__global uint *rowCounters, __global uint *debug)
{
    uint                tid = get_global_id(0);
    ulong               v[16];
    uint                inputs_per_thread = NR_INPUTS / get_global_size(0);
    uint                input = tid * inputs_per_thread;
    uint                input_end = (tid + 1) * inputs_per_thread;
    uint                dropped = 0;
    while (input < input_end)
      {
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
	v[8] =  blake_iv[0];
	v[9] =  blake_iv[1];
	v[10] = blake_iv[2];
	v[11] = blake_iv[3];
	v[12] = blake_iv[4];
	v[13] = blake_iv[5];
	v[14] = blake_iv[6];
	v[15] = blake_iv[7];
	// mix in length of data
	v[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */;
	// last block
	v[14] ^= (ulong)-1;

	// round 1
	mix(v[0], v[4], v[8],  v[12], 0, word1);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 2
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], word1, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 3
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, word1);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 4
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, word1);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 5
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, word1);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 6
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], word1, 0);
	// round 7
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], word1, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 8
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, word1);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 9
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], word1, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 10
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], word1, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 11
	mix(v[0], v[4], v[8],  v[12], 0, word1);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 12
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], word1, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);

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
#if ZCASH_HASH_LEN == 50
	dropped += ht_store(0, ht, input * 2,
		h[0],
		h[1],
		h[2],
		h[3], rowCounters);
	dropped += ht_store(0, ht, input * 2 + 1,
		(h[3] >> 8) | (h[4] << (64 - 8)),
		(h[4] >> 8) | (h[5] << (64 - 8)),
		(h[5] >> 8) | (h[6] << (64 - 8)),
		(h[6] >> 8), rowCounters);
#else
#error "unsupported ZCASH_HASH_LEN"
#endif

	input++;
      }
#ifdef ENABLE_DEBUG
    debug[tid * 2] = 0;
    debug[tid * 2 + 1] = dropped;
#endif
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
__device__ 
uint xor_and_store(uint round, __global char *ht_dst, uint row,
	uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
	__global uint *rowCounters)
{
    ulong xi0, xi1, xi2;
#if NR_ROWS_LOG >= 16 && NR_ROWS_LOG <= 20
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
	xi1 = *(__global uint *)a ^ *(__global uint *)b;
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
#else
#error "unsupported NR_ROWS_LOG"
#endif
    return ht_store(round, ht_dst, ENCODE_INPUTS(row, slot_a, slot_b),
	    xi0, xi1, xi2, 0, rowCounters);
}


/*
** Execute one Equihash round. Read from ht_src, XOR colliding pairs of Xi,
** store them in ht_dst.
*/
__device__
void equihash_round(uint round,
	__global char *ht_src,
	__global char *ht_dst,
	__global uint *debug,
	 uchar *first_words_data,
	 uint *collisionsData,
	 uint *collisionsNum,
	__global uint *rowCountersSrc,
	__global uint *rowCountersDst)
{
    uint		tid = get_global_id(0);
    uint		tlid = get_local_id(0);
    __global char	*p;
    uint		cnt;
    uchar	*first_words = &first_words_data[(NR_SLOTS)*tlid];
    uchar		mask;
    uint		i, j;
    // NR_SLOTS is already oversized (by a factor of OVERHEAD), but we want to
    // make it even larger
#ifdef ENABLE_DEBUG
	uint dropped_coll = 0;
	uint dropped_stor = 0;
#endif
	__global ulong	*a, *b;
    uint		xi_offset;
    // read first words of Xi from the previous (round - 1) hash table
    xi_offset = xi_offset_for_round(round - 1);
    // the mask is also computed to read data from the previous round
#if NR_ROWS_LOG == 16
    mask = ((!(round % 2)) ? 0x0f : 0xf0);
#elif NR_ROWS_LOG == 18
    mask = ((!(round % 2)) ? 0x03 : 0x30);
#elif NR_ROWS_LOG == 19
    mask = ((!(round % 2)) ? 0x01 : 0x10);
#elif NR_ROWS_LOG == 20
    mask = 0; /* we can vastly simplify the code below */
#else
#error "unsupported NR_ROWS_LOG"
#endif
    uint thCollNum = 0;
    *collisionsNum = 0;
    //barrier(CLK_LOCAL_MEM_FENCE);
    p = (ht_src + tid * NR_SLOTS * SLOT_LEN);
    uint rowIdx = tid/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
    cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
    if (!cnt)
	// no elements in row, no collisions
	goto part2;
    p += xi_offset;
    for (i = 0; i < cnt; i++, p += SLOT_LEN)
	first_words[i] = (*(__global uchar *)p) & mask;
    // find collisions
    for (i = 0; i < cnt-1 && thCollNum < COLL_DATA_SIZE_PER_TH; i++)
      {
	uchar data_i = first_words[i];
	uint collision = (tid << 10) | (i << 5) | (i + 1);
	for (j = i+1; (j+4) < cnt;)
	  {
	      {
		uint isColl = ((data_i == first_words[j]) ? 1 : 0);
		if (isColl)
		  {
		    thCollNum++;
		    uint index = atomic_inc(collisionsNum);
		    collisionsData[index] = collision;
		  }
		collision++;
		j++;
	      }
	      {
		uint isColl = ((data_i == first_words[j]) ? 1 : 0);
		if (isColl)
		  {
		    thCollNum++;
		    uint index = atomic_inc(collisionsNum);
		    collisionsData[index] = collision;
		  }
		collision++;
		j++;
	      }
	      {
		uint isColl = ((data_i == first_words[j]) ? 1 : 0);
		if (isColl)
		  {
		    thCollNum++;
		    uint index = atomic_inc(collisionsNum);
		    collisionsData[index] = collision;
		  }
		collision++;
		j++;
	      }
	      {
		uint isColl = ((data_i == first_words[j]) ? 1 : 0);
		if (isColl)
		  {
		    thCollNum++;
		    uint index = atomic_inc(collisionsNum);
		    collisionsData[index] = collision;
		  }
		collision++;
		j++;
	      }
	  }
	for (; j < cnt; j++)
	  {
	    uint isColl = ((data_i == first_words[j]) ? 1 : 0);
	    if (isColl)
	      {
		thCollNum++;
		uint index = atomic_inc(collisionsNum);
		collisionsData[index] = collision;
	      }
	    collision++;
	  }
      }

part2:
    barrier(CLK_LOCAL_MEM_FENCE);
    uint totalCollisions = *collisionsNum;
    for (uint index = tlid; index < totalCollisions; index += get_local_size(0))
      {
	uint collision = collisionsData[index];
	uint collisionThreadId = collision >> 10;
	uint i = (collision >> 5) & 0x1F;
	uint j = collision & 0x1F;
	__global char *ptr = ht_src + collisionThreadId * NR_SLOTS * SLOT_LEN +
	    xi_offset;
	a = (__global ulong *)(ptr + i * SLOT_LEN);
	b = (__global ulong *)(ptr + j * SLOT_LEN);
	dropped_stor += xor_and_store(round, ht_dst, collisionThreadId, i, j,
		a, b, rowCountersDst);
	}
#ifdef ENABLE_DEBUG
	debug[tid * 2] = dropped_coll;
	debug[tid * 2 + 1] = dropped_stor;
#endif
}



/*
** This defines kernel_round1, kernel_round2, ..., kernel_round7.
*/
#define KERNEL_ROUND(N) \
__global__ void kernel_round ## N(__global char *ht_src, __global char *ht_dst, \
	__global uint *rowCountersSrc, __global uint *rowCountersDst, \
       	__global uint *debug) \
{ \
    __local uchar first_words_data[(NR_SLOTS)*THRD]; \
    __local uint    collisionsData[COLL_DATA_SIZE_PER_TH * THRD];\
    __local uint    collisionsNum; \
    equihash_round(N, ht_src, ht_dst, debug, first_words_data, collisionsData, \
	    &collisionsNum, rowCountersSrc, rowCountersDst); \
}
KERNEL_ROUND(1)
KERNEL_ROUND(2)
KERNEL_ROUND(3)
KERNEL_ROUND(4)
KERNEL_ROUND(5)
KERNEL_ROUND(6)
KERNEL_ROUND(7)


// kernel_round8 takes an extra argument, "sols"
__global__ 
void kernel_round8(__global char *ht_src, __global char *ht_dst,
	__global uint *rowCountersSrc, __global uint *rowCountersDst,
	__global uint *debug, __global sols_t *sols)
{
    uint		tid = get_global_id(0);
    __local uchar	first_words_data[(NR_SLOTS)*THRD];
    __local uint    collisionsData[COLL_DATA_SIZE_PER_TH * THRD];
    __local uint	collisionsNum;
    equihash_round(8, ht_src, ht_dst, debug, first_words_data, collisionsData,
	    &collisionsNum, rowCountersSrc, rowCountersDst);
    if (!tid)
	sols->nr = sols->likely_invalids = 0;
}


__device__ 
uint expand_ref(__global char *ht, uint xi_offset, uint row, uint slot)
{
    return *(__global uint *)(ht + row * NR_SLOTS * SLOT_LEN +
	    slot * SLOT_LEN + xi_offset - 4);
}


__device__ 
uint expand_refs(uint *ins, uint nr_inputs, __global char **htabs,
	uint round)
{
    __global char	*ht = htabs[round % 2];
    uint		i = nr_inputs - 1;
    uint		j = nr_inputs * 2 - 1;
    uint		xi_offset = xi_offset_for_round(round);
    int			dup_to_watch = -1;
    do
      {
	ins[j] = expand_ref(ht, xi_offset,
		DECODE_ROW(ins[i]), DECODE_SLOT1(ins[i]));
	ins[j - 1] = expand_ref(ht, xi_offset,
		DECODE_ROW(ins[i]), DECODE_SLOT0(ins[i]));
	if (!round)
	  {
	    if (dup_to_watch == -1)
		dup_to_watch = ins[j];
	    else if (ins[j] == dup_to_watch || ins[j - 1] == dup_to_watch)
		return 0;
	  }
	if (!i)
	    break ;
	i--;
	j -= 2;
      }
    while (1);
    return 1;
}


/*
** Verify if a potential solution is in fact valid.
*/
__device__ 
void potential_sol(__global char **htabs, __global sols_t *sols,
	uint ref0, uint ref1)
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
	if (!expand_refs(values_tmp, nr_values, htabs, round))
	    return ;
	nr_values *= 2;
      }
    while (round > 0);
    // solution appears valid, copy it to sols
    sol_i = atomic_inc(&sols->nr);
    if (sol_i >= MAX_SOLS)
	return ;
    for (i = 0; i < (1 << PARAM_K); i++)
	sols->values[sol_i][i] = values_tmp[i];
    sols->valid[sol_i] = 1;
}


/*
** Scan the hash tables to find Equihash solutions.
*/
__global__
void kernel_sols(__global char *ht0, __global char *ht1, __global sols_t *sols,
	__global uint *rowCountersSrc, __global uint *rowCountersDst)
{
    uint		tid = get_global_id(0);
    __global char	*htabs[2] = { ht0, ht1 };
    __global char	*hcounters[2] = { (char *)rowCountersSrc, (char *)rowCountersDst };
    uint		ht_i = (PARAM_K - 1) % 2; // table filled at last round
    uint		cnt;
    uint		xi_offset = xi_offset_for_round(PARAM_K - 1);
    uint		i, j;
    __global char	*a, *b;
    uint		ref_i, ref_j;
    // it's ok for the collisions array to be so small, as if it fills up
    // the potential solutions are likely invalid (many duplicate inputs)
    ulong		collisions;
#if NR_ROWS_LOG >= 16 && NR_ROWS_LOG <= 20
    // in the final hash table, we are looking for a match on both the bits
    // part of the previous PREFIX colliding bits, and the last PREFIX bits.
    uint		mask = 0xffffff;
#else
#error "unsupported NR_ROWS_LOG"
#endif
    a = htabs[ht_i] + tid * NR_SLOTS * SLOT_LEN;
    uint rowIdx = tid/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
    cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
    a += xi_offset;
    for (i = 0; i < cnt; i++, a += SLOT_LEN)
      {
	uint a_data = ((*(__global uint *)a) & mask);
	ref_i = *(__global uint *)(a - 4);
	for (j = i + 1, b = a + SLOT_LEN; j < cnt; j++, b += SLOT_LEN)
	  {
	    if (a_data == ((*(__global uint *)b) & mask))
	      {
		ref_j = *(__global uint *)(b - 4);
		collisions = ((ulong)ref_i << 32) | ref_j;
		goto exit1;
	      }
	  }
      }
    return;

exit1:
    potential_sol(htabs, sols, collisions >> 32, collisions & 0xffffffff);

}

struct __align__(64) c_context {
	char* buf_ht[2], *buf_sols, *buf_dbg;
	char *rowCounters[2];
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
	checkCudaErrors(cudaMalloc((void**)&eq->rowCounters[0], NR_ROWS));
	checkCudaErrors(cudaMalloc((void**)&eq->rowCounters[1], NR_ROWS));

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
	//printf("NR_SLOTS=%d NR_ROWS=%d\n",NR_SLOTS,NR_ROWS);
	c_context *miner = eq;
	
	//FUNCTION<<<totalblocks, threadsperblock>>>(ARGUMENTS)

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)context, 128, 0);

	void* buf_blake_st;
	checkCudaErrors(cudaMalloc((void**)&buf_blake_st, sizeof(blake2b_state_s)));
	checkCudaErrors(cudaMemcpy(buf_blake_st, &initialCtx, sizeof(blake2b_state_s), cudaMemcpyHostToDevice));
	
	for (unsigned round = 0; round < PARAM_K; round++) {
//		if (round < 2) {
			//every round
			checkCudaErrors(cudaMemset(miner->rowCounters[round % 2],0,NR_ROWS));
//			kernel_init_ht<<<NR_ROWS / ROWS_PER_UINT / 256, 256>>>(miner->buf_ht[round & 1],(uint*)miner->rowCounters[round % 2]);
//			printf("%d %d %d\n",NR_ROWS / ROWS_PER_UINT / 256, NR_ROWS,ROWS_PER_UINT);
//			exit(-1);
//		}
		if (!round)	{
			miner->global_ws = select_work_size_blake();
		} else {
			miner->global_ws = NR_ROWS;
		}
		// cancel function
		switch (round) {
		case 0:
			kernel_round0<<<NR_INPUTS/THRD,THRD>>>((ulong*)buf_blake_st, miner->buf_ht[round & 1], (uint*)miner->rowCounters[round % 2],(uint*)miner->buf_dbg);
//			printf("%d %d\n",totalblocks, NR_INPUTS);
//                        exit(-1);
			break;
		case 1:
			kernel_round1<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
			//exit(0);
			break;
		case 2:
			kernel_round2<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
//			exit(0);
			break;
		case 3:
			kernel_round3<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
			break;
		case 4:
			kernel_round4<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
			break;
		case 5:
			kernel_round5<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
//exit(0);
			break;
		case 6:
			kernel_round6<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
	//	exit(0);
			break;
		case 7:
			kernel_round7<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
//			exit(0);
			break;
		case 8:
			kernel_round8<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg, (sols_t*)miner->buf_sols);
			break;
		}
		if (cancelf()) return;
	}
	kernel_sols<<<NR_ROWS/32,32>>>(miner->buf_ht[0], miner->buf_ht[1], (sols_t*)miner->buf_sols,(uint*)miner->rowCounters[0],(uint*)miner->rowCounters[1]);

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
