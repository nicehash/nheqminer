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

#ifndef _WIN32
#include <cpuid.h>
#else
#include <intrin.h>
#endif // !WIN32

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <boost/thread.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
typedef unsigned char u_char;
#endif

enum {
    // Verus Key size must include the equivalent size of a Haraka key
    // after the first part.
    // Any excess over a power of 2 will not get mutated, and any excess over
    // power of 2 + Haraka sized key will not be used
    VERUSKEYSIZE=1024 * 8 + (40 * 16),
    VERUSHHASH_SOLUTION_VERSION = 1
};

struct verusclhash_descr
{
    uint256 seed;
    uint32_t keySizeInBytes;
};

struct thread_specific_ptr {
    void *ptr;
    thread_specific_ptr() { ptr = NULL; }
    void reset(void *newptr = NULL)
    {
        if (ptr && ptr != newptr)
        {
            std::free(ptr);
        }
        ptr = newptr;
    }
    void *get() { return ptr; }
#ifdef _WIN32 // horrible MingW and gcc thread local storage bug workaround
    ~thread_specific_ptr();
#else
    ~thread_specific_ptr() {
        this->reset();
    }
#endif
};

extern thread_local thread_specific_ptr verusclhasher_key;
extern thread_local thread_specific_ptr verusclhasher_descr;

extern int __cpuverusoptimized;

inline bool IsCPUVerusOptimized()
{
    if (__cpuverusoptimized & 0x80)
    {
#ifdef _WIN32
        #define bit_AVX		(1 << 28)
        #define bit_AES		(1 << 25)
        #define bit_PCLMUL  (1 << 1)
        // https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
        // bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;

        int cpuInfo[4];
		__cpuid(cpuInfo, 1);
        __cpuverusoptimized = ((cpuInfo[2] & (bit_AVX | bit_AES | bit_PCLMUL)) == (bit_AVX | bit_AES | bit_PCLMUL));
#else
        unsigned int eax,ebx,ecx,edx;

        if (!__get_cpuid(1,&eax,&ebx,&ecx,&edx))
        {
            __cpuverusoptimized = false;
        }
        else
        {
            __cpuverusoptimized = ((ecx & (bit_AVX | bit_AES | bit_PCLMUL)) == (bit_AVX | bit_AES | bit_PCLMUL));
        }
#endif //WIN32
    }
    return __cpuverusoptimized;
};

inline void ForceCPUVerusOptimized(bool trueorfalse)
{
    __cpuverusoptimized = trueorfalse;
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
    uint64_t keySizeInBytes;
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

    // align on 256 bit boundary at end
    verusclhasher(uint64_t keysize=VERUSKEYSIZE) : keySizeInBytes((keysize >> 5) << 5)
    {
        if (IsCPUVerusOptimized())
        {
            verusclhashfunction = &verusclhash;
        }
        else
        {
            verusclhashfunction = &verusclhash_port;
        }

        // if we changed, change it
        if (verusclhasher_key.get() && keySizeInBytes != ((verusclhash_descr *)verusclhasher_descr.get())->keySizeInBytes)
        {
            verusclhasher_key.reset();
            verusclhasher_descr.reset();
        }
        // get buffer space for mutating and refresh keys
        void *key = NULL;
        if (!(key = verusclhasher_key.get()) && 
            (verusclhasher_key.reset((unsigned char *)alloc_aligned_buffer(keySizeInBytes << 1)), key = verusclhasher_key.get()))
        {
            verusclhash_descr *pdesc;
            if (verusclhasher_descr.reset(new verusclhash_descr()), pdesc = (verusclhash_descr *)verusclhasher_descr.get())
            {
                pdesc->keySizeInBytes = keySizeInBytes;
            }
            else
            {
                verusclhasher_key.reset();
                key = NULL;
            }
        }
        if (key)
        {
            keyMask = keymask(keySizeInBytes);
        }
        else
        {
            keyMask = 0;
            keySizeInBytes = 0;
        }
#ifdef VERUSHASHDEBUG
        printf("New hasher, keyMask: %lx, newKeySize: %lx\n", keyMask, keySizeInBytes);
#endif
    }

    // this prepares a key for hashing and mutation by copying it from the original key for this block
    // WARNING!! this does not check for NULL ptr, so make sure the buffer is allocated
    inline void *gethashkey()
    {
        unsigned char *ret = (unsigned char *)verusclhasher_key.get();
        verusclhash_descr *pdesc = (verusclhash_descr *)verusclhasher_descr.get();
        memcpy(ret, ret + pdesc->keySizeInBytes, keyMask + 1);
#ifdef VERUSHASHDEBUG
        // in debug mode, ensure that what should be the same, is
        assert(memcmp(ret + (keyMask + 1), ret + (pdesc->keySizeInBytes + keyMask + 1), verusclhasher_keySizeInBytes - (keyMask + 1)) == 0);
#endif
        return ret;
    }

    inline void *gethasherrefresh()
    {
        verusclhash_descr *pdesc = (verusclhash_descr *)verusclhasher_descr.get();
        return (unsigned char *)verusclhasher_key.get() + pdesc->keySizeInBytes;
    }

    inline verusclhash_descr *gethasherdescription()
    {
        return (verusclhash_descr *)verusclhasher_descr.get();
    }

    inline uint64_t keyrefreshsize()
    {
        return keyMask + 1;
    }

    inline uint64_t operator()(const unsigned char buf[64]) const {
        return (*verusclhashfunction)(verusclhasher_key.get(), buf, keyMask);
    }

    inline uint64_t operator()(const unsigned char buf[64], void *key) const {
        return (*verusclhashfunction)(key, buf, keyMask);
    }
};

#endif // #ifdef __cplusplus

#endif // INCLUDE_VERUS_CLHASH_H
