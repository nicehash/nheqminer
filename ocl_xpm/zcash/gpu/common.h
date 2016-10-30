#if defined(__OPENCL_HOST__)
#define __global
//#include "blake2/blake2.h"
//#include "equi.h"
#include "../cpu_tromp/equi.h"

#else
typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;

#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__ ((__aligned__(x)))
#endif

enum blake2b_constant
{
  BLAKE2B_BLOCKBYTES = 128,
  BLAKE2B_OUTBYTES   = 64,
  BLAKE2B_KEYBYTES   = 64,
  BLAKE2B_SALTBYTES  = 16,
  BLAKE2B_PERSONALBYTES = 16
};

#pragma pack(push, 1)
ALIGN( 64 ) typedef struct __blake2b_state {
  uint64_t h[8];
  uint8_t  buf[BLAKE2B_BLOCKBYTES];
  uint16_t counter;
  uint8_t  buflen;
  uint8_t  lastblock;
} blake2b_state;
#pragma pack(pop)
#endif

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))


#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
//#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))
//#define BASE (1u<<DIGITBITS)
//#define NHASHES (2u*BASE)
//#define HASHESPERBLAKE (512/WN)
//#define HASHOUT (HASHESPERBLAKE*WN/8)

// 2_log of number of buckets
#define BUCKBITS  (DIGITBITS-RESTBITS)

// number of buckets
#define NBUCKETS (1<<BUCKBITS)
// 2_log of number of slots per bucket
#define SLOTBITS (RESTBITS+1+1)
// number of slots per bucket
#define NSLOTS (1u<<SLOTBITS)
// number of per-xhash slots
#define XFULL 16
// SLOTBITS mask
#define SLOTMASK (NSLOTS-1)
// number of possible values of xhash (rest of n) bits
#define NRESTS (1u<<RESTBITS)
// number of blocks of hashes extracted from single 512 bit blake2b output
#define NBLOCKS ((NHASHES+HASHESPERBLAKE-1)/HASHESPERBLAKE)
// nothing larger found in 100000 runs
#define MAXSOLS 8

#define WORDS(bits)     ((bits + 31) / 32)
#define HASHWORDS0 WORDS(WN - DIGITBITS + RESTBITS)
#define HASHWORDS1 WORDS(WN - 2*DIGITBITS + RESTBITS)

typedef uint32_t proof[PROOFSIZE];

// tree  = | xhash(RESTBITS)    | slotid1(SLOTBITS) | slotid0(SLOTBITS) | bucketid(BUCKBITS) |
// index = | bucketid(BUCKBITS) | slotid0(SLOTBITS) |
typedef uint32_t tree;

typedef union hashunit {
  uint32_t word;
  uint8_t bytes[4];
} hashunit;

typedef struct slot0 {
  tree attr;
  hashunit hash[HASHWORDS0];
} slot0;

typedef struct slot1 {
  tree attr;
  hashunit hash[HASHWORDS1];
} slot1;

// a bucket is NSLOTS treenodes
typedef slot0 bucket0[NSLOTS];
typedef slot1 bucket1[NSLOTS];
// the N-bit hash consists of K+1 n-bit "digits"
// each of which corresponds to a layer of NBUCKETS buckets
typedef bucket0 digit0[NBUCKETS];
typedef bucket1 digit1[NBUCKETS];

// manages hash and tree data
typedef struct htalloc {
  __global bucket0 *trees0[(WK+1)/2];
  __global bucket1 *trees1[WK/2];
} htalloc;

typedef uint32_t bsizes[NBUCKETS];


typedef struct htlayout {
  htalloc hta;
  uint32_t prevhashunits;
  uint32_t nexthashunits;
  uint32_t dunits;
  uint32_t prevbo;
  uint32_t nextbo;
} htlayout;

#if RESTBITS <= 6
  typedef uint8_t xslot;
#else
  typedef uint16_t xslot;
#endif

typedef struct collisiondata {
#ifdef XBITMAP
#if NSLOTS > 64
#error cant use XBITMAP with more than 64 slots
#endif
  uint64_t xhashmap[NRESTS];
  uint64_t xmap;
#else
  xslot nxhashslots[NRESTS];
  xslot xhashslots[NRESTS][XFULL];
  xslot *xx;
  uint32_t n0;
  uint32_t n1;
#endif
  uint32_t s0;
} collisiondata;


typedef struct equi {
  blake2b_state blake_ctx;
  htalloc hta;
  __global bsizes *nslots;
  __global proof *sols;
  uint32_t nsols;
  uint32_t nthreads;
} equi;
