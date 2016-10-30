# 1 "input.cl"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "input.cl"
# 1 "param.h" 1
# 60 "param.h"
typedef struct sols_s
{
    uint nr;
    uint likely_invalidss;
    uchar valid[2000];
    uint values[2000][(1 << 9)];
} sols_t;
# 2 "input.cl" 2
# 35 "input.cl"
__constant ulong blake_iv[] =
{
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};




__kernel
void kernel_init_ht(__global char *ht)
{
    uint tid = get_global_id(0);
    *(__global uint *)(ht + tid * ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 32) = 0;
}
# 79 "input.cl"
uint ht_store(uint round, __global char *ht, uint i,
        ulong xi0, ulong xi1, ulong xi2, ulong xi3)
{
    uint row;
    __global char *p;
    uint cnt;

    if (!(round % 2))
 row = (xi0 & 0xffff);
    else




 row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
     ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
# 119 "input.cl"
    xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
    xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
    xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
    p = ht + row * ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 32;
    cnt = atomic_inc((__global uint *)p);
    if (cnt >= ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3))
        return 1;
    p += cnt * 32 + (8 + ((round) / 2) * 4);

    *(__global uint *)(p - 4) = i;
    if (round == 0 || round == 1)
      {

 *(__global ulong *)(p + 0) = xi0;
 *(__global ulong *)(p + 8) = xi1;
 *(__global ulong *)(p + 16) = xi2;
      }
    else if (round == 2)
      {

 *(__global ulong *)(p + 0) = xi0;
 *(__global ulong *)(p + 8) = xi1;
 *(__global uint *)(p + 16) = xi2;
      }
    else if (round == 3 || round == 4)
      {

 *(__global ulong *)(p + 0) = xi0;
 *(__global ulong *)(p + 8) = xi1;

      }
    else if (round == 5)
      {

 *(__global ulong *)(p + 0) = xi0;
 *(__global uint *)(p + 8) = xi1;
      }
    else if (round == 6 || round == 7)
      {

 *(__global ulong *)(p + 0) = xi0;
      }
    else if (round == 8)
      {

 *(__global uint *)(p + 0) = xi0;
      }
    return 0;
}
# 187 "input.cl"
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void kernel_round0(__global ulong *blake_state, __global char *ht,
        __global uint *debug)
{
    uint tid = get_global_id(0);
    ulong v[16];
    uint inputs_per_thread = (1 << (200 / (9 + 1))) / get_global_size(0);
    uint input = tid * inputs_per_thread;
    uint input_end = (tid + 1) * inputs_per_thread;
    uint dropped = 0;
    while (input < input_end)
      {


        ulong word1 = (ulong)input << 32;

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

        v[12] ^= 140 + 4 ;

        v[14] ^= -1;


        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + word1); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + word1); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + word1); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + word1); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + word1); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + word1); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + word1); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + word1); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + word1); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + word1); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + word1); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;

        v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 32); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 24); v[0] = (v[0] + v[4] + 0); v[12] = rotate((v[12] ^ v[0]), (ulong)64 - 16); v[8] = (v[8] + v[12]); v[4] = rotate((v[4] ^ v[8]), (ulong)64 - 63);;
        v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 32); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 24); v[1] = (v[1] + v[5] + 0); v[13] = rotate((v[13] ^ v[1]), (ulong)64 - 16); v[9] = (v[9] + v[13]); v[5] = rotate((v[5] ^ v[9]), (ulong)64 - 63);;
        v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 32); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 24); v[2] = (v[2] + v[6] + 0); v[14] = rotate((v[14] ^ v[2]), (ulong)64 - 16); v[10] = (v[10] + v[14]); v[6] = rotate((v[6] ^ v[10]), (ulong)64 - 63);;
        v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 32); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 24); v[3] = (v[3] + v[7] + 0); v[15] = rotate((v[15] ^ v[3]), (ulong)64 - 16); v[11] = (v[11] + v[15]); v[7] = rotate((v[7] ^ v[11]), (ulong)64 - 63);;
        v[0] = (v[0] + v[5] + word1); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 32); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 24); v[0] = (v[0] + v[5] + 0); v[15] = rotate((v[15] ^ v[0]), (ulong)64 - 16); v[10] = (v[10] + v[15]); v[5] = rotate((v[5] ^ v[10]), (ulong)64 - 63);;
        v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 32); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 24); v[1] = (v[1] + v[6] + 0); v[12] = rotate((v[12] ^ v[1]), (ulong)64 - 16); v[11] = (v[11] + v[12]); v[6] = rotate((v[6] ^ v[11]), (ulong)64 - 63);;
        v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 32); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 24); v[2] = (v[2] + v[7] + 0); v[13] = rotate((v[13] ^ v[2]), (ulong)64 - 16); v[8] = (v[8] + v[13]); v[7] = rotate((v[7] ^ v[8]), (ulong)64 - 63);;
        v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 32); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 24); v[3] = (v[3] + v[4] + 0); v[14] = rotate((v[14] ^ v[3]), (ulong)64 - 16); v[9] = (v[9] + v[14]); v[4] = rotate((v[4] ^ v[9]), (ulong)64 - 63);;



        ulong h[7];
        h[0] = blake_state[0] ^ v[0] ^ v[8];
        h[1] = blake_state[1] ^ v[1] ^ v[9];
        h[2] = blake_state[2] ^ v[2] ^ v[10];
        h[3] = blake_state[3] ^ v[3] ^ v[11];
        h[4] = blake_state[4] ^ v[4] ^ v[12];
        h[5] = blake_state[5] ^ v[5] ^ v[13];
        h[6] = (blake_state[6] ^ v[6] ^ v[14]) & 0xffff;



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
# 409 "input.cl"
uint xor_and_store(uint round, __global char *ht_dst, uint row,
 uint slot_a, uint slot_b, __global ulong *a, __global ulong *b)
{
    ulong xi0, xi1, xi2;



    if (round == 1 || round == 2)
      {


 xi0 = *(a++) ^ *(b++);
 xi1 = *(a++) ^ *(b++);
 xi2 = *a ^ *b;
      }
    else if (round == 3)
      {

 xi0 = *a++ ^ *b++;
 xi1 = *a++ ^ *b++;
 xi2 = *(__global uint *)a ^ *(__global uint *)b;
      }
    else if (round == 4 || round == 5)
      {

 xi0 = *a++ ^ *b++;
 xi1 = *a ^ *b;
 xi2 = 0;
      }
    else if (round == 6)
      {

 xi0 = *a++ ^ *b++;
 xi1 = *(__global uint *)a ^ *(__global uint *)b;
 xi2 = 0;
      }
    else if (round == 7 || round == 8)
      {

 xi0 = *a ^ *b;
 xi1 = 0;
 xi2 = 0;
      }


    if (!xi0 && !xi1)
 return 0;



    return ht_store(round, ht_dst, ((row << 16) | ((slot_b & 0xff) << 8) | (slot_a & 0xff)),
     xi0, xi1, xi2, 0);
}





void equihash_round(uint round, __global char *ht_src, __global char *ht_dst,
 __global uint *debug)
{
    uint tid = get_global_id(0);
    uint tlid = get_local_id(0);
    __global char *p;
    uint cnt;
    uchar first_words[((1 << (((200 / (9 + 1)) + 1) - 16)) * 3)];
    uchar mask;
    uint i, j;


    ushort collisions[((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 2];
    uint nr_coll = 0;
    uint n;
    uint dropped_coll, dropped_stor;
    __global ulong *a, *b;
    uint xi_offset;

    xi_offset = (8 + ((round - 1) / 2) * 4);


    mask = ((!(round % 2)) ? 0x0f : 0xf0);
# 499 "input.cl"
    p = (ht_src + tid * ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 32);
    cnt = *(__global uint *)p;
    cnt = min(cnt, (uint)((1 << (((200 / (9 + 1)) + 1) - 16)) * 3));
    p += xi_offset;
    for (i = 0; i < cnt; i++, p += 32)
        first_words[i] = *(__global uchar *)p;

    nr_coll = 0;
    dropped_coll = 0;
    for (i = 0; i < cnt; i++)
        for (j = i + 1; j < cnt; j++)
            if ((first_words[i] & mask) ==
      (first_words[j] & mask))
              {

                if (nr_coll >= sizeof (collisions) / sizeof (*collisions))
                    dropped_coll++;
                else


                    collisions[nr_coll++] =
   ((ushort)j << 8) | ((ushort)i & 0xff);



              }

    uint adj = (!(round % 2)) ? 1 : 0;

    dropped_stor = 0;
    for (n = 0; n < nr_coll; n++)
      {
        i = collisions[n] & 0xff;
        j = collisions[n] >> 8;
        a = (__global ulong *)
            (ht_src + tid * ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 32 + i * 32 + xi_offset
      + adj);
        b = (__global ulong *)
            (ht_src + tid * ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 32 + j * 32 + xi_offset
      + adj);
 dropped_stor += xor_and_store(round, ht_dst, tid, i, j, a, b);
      }




}
# 557 "input.cl"
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round1(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(1, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round2(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(2, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round3(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(3, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round4(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(4, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round5(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(5, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round6(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(6, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round7(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(7, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round8(__global char *ht_src, __global char *ht_dst, __global uint *debug) { equihash_round(8, ht_src, ht_dst, debug); }

uint expand_ref(__global char *ht, uint xi_offset, uint row, uint slot)
{
    return *(__global uint *)(ht + row * ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 32 +
     slot * 32 + xi_offset - 4);
}

void expand_refs(__global uint *ins, uint nr_inputs, __global char **htabs,
 uint round)
{
    __global char *ht = htabs[round % 2];
    uint i = nr_inputs - 1;
    uint j = nr_inputs * 2 - 1;
    uint xi_offset = (8 + ((round) / 2) * 4);
    do
      {
 ins[j] = expand_ref(ht, xi_offset,
  (ins[i] >> 16), ((ins[i] >> 8) & 0xff));
 ins[j - 1] = expand_ref(ht, xi_offset,
  (ins[i] >> 16), (ins[i] & 0xff));
 if (!i)
     break ;
 i--;
 j -= 2;
      }
    while (1);
}




void potential_sol(__global char **htabs, __global sols_t *sols,
 uint ref0, uint ref1)
{
    uint sol_i;
    uint nr_values;
    sol_i = atomic_inc(&sols->nr);
    if (sol_i >= 2000)
 return ;
    sols->valid[sol_i] = 0;
    nr_values = 0;
    sols->values[sol_i][nr_values++] = ref0;
    sols->values[sol_i][nr_values++] = ref1;
    uint round = 9 - 1;
    do
      {
 round--;
 expand_refs(&(sols->values[sol_i][0]), nr_values, htabs, round);
 nr_values *= 2;
      }
    while (round > 0);
    sols->valid[sol_i] = 1;
}




__kernel
void kernel_sols(__global char *ht0, __global char *ht1, __global sols_t *sols)
{
    uint tid = get_global_id(0);
    __global char *htabs[2] = { ht0, ht1 };
    uint ht_i = (9 - 1) % 2;
    uint cnt;
    uint xi_offset = (8 + ((9 - 1) / 2) * 4);
    uint i, j;
    __global char *a, *b;
    uint ref_i, ref_j;


    ulong collisions[5];
    uint coll;



    uint mask = 0xffffff;



    if (tid == 0)
 sols->nr = sols->likely_invalidss = 0;
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    a = htabs[ht_i] + tid * ((1 << (((200 / (9 + 1)) + 1) - 16)) * 3) * 32;
    cnt = *(__global uint *)a;
    cnt = min(cnt, (uint)((1 << (((200 / (9 + 1)) + 1) - 16)) * 3));
    coll = 0;
    a += xi_offset;
    for (i = 0; i < cnt; i++, a += 32)
 for (j = i + 1, b = a + 32; j < cnt; j++, b += 32)
     if (((*(__global uint *)a) & mask) ==
      ((*(__global uint *)b) & mask))
       {
  ref_i = *(__global uint *)(a - 4);
  ref_j = *(__global uint *)(b - 4);
  if (coll < sizeof (collisions) / sizeof (*collisions))
      collisions[coll++] = ((ulong)ref_i << 32) | ref_j;
  else
      atomic_inc(&sols->likely_invalidss);
       }
    if (!coll)
 return ;
    for (i = 0; i < coll; i++)
 potential_sol(htabs, sols, collisions[i] >> 32,
  collisions[i] & 0xffffffff);
}
