// (C) 2018 Michael Toutonghi
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/*
This provides the PoW hash function for Verus, enabling CPU mining.
*/
#ifndef VERUS_HASH_H_
#define VERUS_HASH_H_

#include <cstring>
#include <vector>

#include <cpuid.h>

#include "uint256.h"
#include "crypto/verus_clhash.h"

extern "C" 
{
#include "crypto/haraka.h"
#include "crypto/haraka_portable.h"
}

class CVerusHash
{
    public:
        static void Hash(void *result, const void *data, size_t len);
        static void (*haraka512Function)(unsigned char *out, const unsigned char *in);

        static void init();

        CVerusHash() { }

        CVerusHash &Write(const unsigned char *data, size_t len);

        CVerusHash &Reset()
        {
            curBuf = buf1;
            result = buf2;
            curPos = 0;
            std::fill(buf1, buf1 + sizeof(buf1), 0);
            return *this;
        }

        int64_t *ExtraI64Ptr() { return (int64_t *)(curBuf + 32); }
        void ClearExtra()
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
            }
        }
        void ExtraHash(unsigned char hash[32]) { (*haraka512Function)(hash, curBuf); }

        void Finalize(unsigned char hash[32])
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
                (*haraka512Function)(hash, curBuf);
            }
            else
                std::memcpy(hash, curBuf, 32);
        }

    private:
        // only buf1, the first source, needs to be zero initialized
        unsigned char buf1[64] = {0}, buf2[64];
        unsigned char *curBuf = buf1, *result = buf2;
        size_t curPos = 0;
};

class CVerusHashV2
{
    public:
        static void Hash(void *result, const void *data, size_t len);
        static void (*haraka512Function)(unsigned char *out, const unsigned char *in);
        static void (*haraka512KeyedFunction)(unsigned char *out, const unsigned char *in, const u128 *rc);
        static void (*haraka256Function)(unsigned char *out, const unsigned char *in);

        static void init();

        verusclhasher vclh;

        CVerusHashV2() : vclh() {
            // we must have allocated key space, or can't run
            if (verusclhasher_keySizeInBytes == 0)
            {
                printf("ERROR: failed to allocate hash buffer - terminating\n");
                assert(false);
            }
        }

        CVerusHashV2 &Write(const unsigned char *data, size_t len);

        CVerusHashV2 &Reset()
        {
            curBuf = buf1;
            result = buf2;
            curPos = 0;
            std::fill(buf1, buf1 + sizeof(buf1), 0);
        }

        inline int64_t *ExtraI64Ptr() { return (int64_t *)(curBuf + 32); }
        inline void ClearExtra()
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
            }
        }

        template <typename T>
        void FillExtra(const T *_data)
        {
            int len = sizeof(T);
            unsigned char *data = (unsigned char *)_data;
            int pos = curPos;
            int left = 32 - pos;
            do
            {
                std::memcpy(curBuf + 32 + pos, data, left > len ? len : left);
                pos += len;
                left -= len;
            } while (left > 0);
        }
        inline void ExtraHash(unsigned char hash[32]) { (*haraka512Function)(hash, curBuf); }
        inline void ExtraHashKeyed(unsigned char hash[32], u128 *key) { (*haraka512KeyedFunction)(hash, curBuf, key); }

        void Finalize(unsigned char hash[32])
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
                (*haraka512Function)(hash, curBuf);
            }
            else
                std::memcpy(hash, curBuf, 32);
        }

        // chains Haraka256 from 32 bytes to fill the key
        u128 *GenNewCLKey(unsigned char *seedBytes32)
        {
            // skip keygen if it is the current key
            if (verusclhasher_seed != *((uint256 *)seedBytes32))
            {
                // generate a new key by chain hashing with Haraka256 from the last curbuf
                int n256blks = verusclhasher_keySizeInBytes >> 5;
                int nbytesExtra = verusclhasher_keySizeInBytes & 0x1f;
                unsigned char *pkey = (unsigned char *)verusclhasherrefresh;
                unsigned char *psrc = seedBytes32;
                for (int i = 0; i < n256blks; i++)
                {
                    (*haraka256Function)(pkey, psrc);
                    psrc = pkey;
                    pkey += 32;
                }
                if (nbytesExtra)
                {
                    unsigned char buf[32];
                    (*haraka256Function)(buf, psrc);
                    memcpy(pkey, buf, nbytesExtra);
                }
                verusclhasher_seed = *((uint256 *)seedBytes32);
            }
            memcpy(verusclhasher_random_data_, verusclhasherrefresh, vclh.keySizeIn64BitWords << 3);
            return (u128 *)verusclhasher_random_data_;
        }

        inline uint64_t IntermediateTo128Offset(uint64_t intermediate)
        {
            // the mask is where we wrap
            uint64_t mask = vclh.keyMask >> 4;
            uint64_t offset = intermediate & mask;
            int64_t wrap = (offset + 39) - mask;
            return wrap > 0 ? wrap : offset;
        }

        void Finalize2b(unsigned char hash[32])
        {
            ClearExtra();

            //uint256 *bhalf1 = (uint256 *)curBuf;
            //uint256 *bhalf2 = bhalf1 + 1;
            //printf("Curbuf: %s%s\n", bhalf1->GetHex().c_str(), bhalf2->GetHex().c_str());

            // gen new key with what is last in buffer
            GenNewCLKey(curBuf);

            // run verusclhash on the buffer
            uint64_t intermediate = vclh(curBuf);

            //printf("intermediate %lx\n", intermediate);

            // fill buffer to the end with the result
            FillExtra(&intermediate);

            //printf("Curbuf: %s%s\n", bhalf1->GetHex().c_str(), bhalf2->GetHex().c_str());

            // get the final hash with a mutated dynamic key for each hash result
            (*haraka512KeyedFunction)(hash, curBuf, (u128 *)verusclhasher_random_data_ + IntermediateTo128Offset(intermediate));
        }

        inline unsigned char *CurBuffer()
        {
            return curBuf;
        }

    private:
        // only buf1, the first source, needs to be zero initialized
        alignas(16) unsigned char buf1[64] = {0}, buf2[64];
        unsigned char *curBuf = buf1, *result = buf2;
        size_t curPos = 0;
};

extern void verus_hash(void *result, const void *data, size_t len);
extern void verus_hash_v2(void *result, const void *data, size_t len);

inline bool IsCPUVerusOptimized()
{
    unsigned int eax,ebx,ecx,edx;

    if (!__get_cpuid(1,&eax,&ebx,&ecx,&edx))
    {
        return false;
    }
    return ((ecx & (bit_AVX | bit_AES | bit_PCLMUL)) == (bit_AVX | bit_AES | bit_PCLMUL));
};

#endif
