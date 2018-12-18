/*
 * This uses veriations of the clhash algorithm for Verus Coin, licensed
 * with the Apache-2.0 open source license.
 * 
 * Copyright (c) 2018 Michael Toutonghi
 * Distributed under the Apache 2.0 software license, available in the original form for clhash
 * here: https://github.com/lemire/clhash/commit/934da700a2a54d8202929a826e2763831bd43cf7#diff-9879d6db96fd29134fc802214163b95a
 * 
 * CLHash is a very fast hashing function that uses the
 * carry-less multiplication and SSE instructions.
 *
 * Original CLHash code (C) 2017, 2018 Daniel Lemire and Owen Kaser
 * Faster 64-bit universal hashing
 * using carry-less multiplications, Journal of Cryptographic Engineering (to appear)
 *
 * Best used on recent x64 processors (Haswell or better).
 *
 **/

#ifndef INCLUDE_VERUS_CLHASH_H
#define INCLUDE_VERUS_CLHASH_H

#include <cpuid.h>

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __WIN32
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
typedef unsigned char u_char
#endif

enum {
    // Verus Key size must include the equivalent size of a Haraka key
    // after the first part.
    // Any excess over a power of 2 will not get mutated, and any excess over
    // power of 2 + Haraka sized key will not be used
    VERUSKEYSIZE=1024 * 8 + (40 * 16)
};

extern thread_local void *verusclhasher_random_data_;
extern thread_local void *verusclhasherrefresh;
extern thread_local int64_t verusclhasher_keySizeInBytes;
extern thread_local uint256 verusclhasher_seed;

static int __cpuverusoptimized = 0x80;

inline bool IsCPUVerusOptimized()
{
    if (__cpuverusoptimized & 0x80)
    {
        unsigned int eax,ebx,ecx,edx;

        if (!__get_cpuid(1,&eax,&ebx,&ecx,&edx))
        {
            __cpuverusoptimized = false;
        }
        else
        {
            __cpuverusoptimized = ((ecx & (bit_AVX | bit_AES | bit_PCLMUL)) == (bit_AVX | bit_AES | bit_PCLMUL));
        }
    }
    return __cpuverusoptimized;
};

uint64_t verusclhash(void * random, const unsigned char buf[64], uint64_t keyMask);
uint64_t verusclhash_port(void * random, const unsigned char buf[64], uint64_t keyMask);

void *alloc_aligned_buffer(uint64_t bufSize);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus

#include <vector>
#include <string>

// special high speed hasher for VerusHash 2.0
struct verusclhasher {
    uint64_t keySizeIn64BitWords;
    uint64_t keyMask;
    uint64_t (*verusclhashfunction)(void * random, const unsigned char buf[64], uint64_t keyMask);

    inline uint64_t keymask(uint64_t keysize)
    {
        int i = 0;
        while (keysize >>= 1)
        {
            i++;
        }
        return i ? (((uint64_t)1) << i) - 1 : 0;
    }

    // align on 128 byte boundary at end
    verusclhasher(uint64_t keysize=VERUSKEYSIZE) : keySizeIn64BitWords((keysize >> 5) << 2)
    {
        if (IsCPUVerusOptimized())
        {
            verusclhashfunction = &verusclhash;
        }
        else
        {
            verusclhashfunction = &verusclhash_port;
        }

        // align to 128 bits
        uint64_t newKeySize = keySizeIn64BitWords << 3;
        if (verusclhasher_random_data_ && newKeySize != verusclhasher_keySizeInBytes)
        {
            freehashkey();
        }
        // get buffer space for 2 keys
        if (verusclhasher_random_data_ || (verusclhasher_random_data_ = alloc_aligned_buffer(newKeySize << 1)))
        {
            verusclhasherrefresh = ((char *)verusclhasher_random_data_) + newKeySize;
            verusclhasher_keySizeInBytes = newKeySize;
            keyMask = keymask(newKeySize);
        }
#ifdef VERUSHASHDEBUG
        printf("New hasher, keyMask: %lx, newKeySize: %lx, keySizeIn64BitWords: %lx\n", keyMask, newKeySize, keySizeIn64BitWords);
#endif
    }

    void freehashkey()
    {
        // no chance for double free
        if (verusclhasher_random_data_)
        {
            std::free((void *)verusclhasher_random_data_);
            verusclhasher_random_data_ = NULL;
            verusclhasherrefresh = NULL;
        }
        verusclhasher_keySizeInBytes = 0;
        keySizeIn64BitWords = 0;
        keyMask = 0;
    }

    // this prepares a key for hashing and mutation by copying it from the original key for this block
    // WARNING!! this does not check for NULL ptr, so make sure the buffer is allocated
    inline void *gethashkey()
    {
        memcpy(verusclhasher_random_data_, verusclhasherrefresh, keyMask + 1);
#ifdef VERUSHASHDEBUG
        // in debug mode, ensure that what should be the same, is
        assert(memcmp((unsigned char *)verusclhasher_random_data_ + (keyMask + 1), (unsigned char *)verusclhasherrefresh + (keyMask + 1), verusclhasher_keySizeInBytes - (keyMask + 1)) == 0);
#endif
        return verusclhasher_random_data_;
    }

    inline uint64_t operator()(const unsigned char buf[64]) const {
        return (*verusclhashfunction)(verusclhasher_random_data_, buf, keyMask);
    }
};

#endif // #ifdef __cplusplus

#endif // INCLUDE_VERUS_CLHASH_H
