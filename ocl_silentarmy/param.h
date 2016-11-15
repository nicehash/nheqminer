#define PARAM_N				200
#define PARAM_K				9
#define PREFIX                          (PARAM_N / (PARAM_K + 1))
#define NR_INPUTS                       (1 << PREFIX)
// Approximate log base 2 of number of elements in hash tables
#define APX_NR_ELMS_LOG                 (PREFIX + 1)
// Number of rows and slots is affected by this; 20 offers the best performance
#define NR_ROWS_LOG                     20

// Setting this to 1 might make SILENTARMY faster, see TROUBLESHOOTING.md
#define OPTIM_SIMPLIFY_ROUND		1

// Number of collision items to track, per thread
#define COLL_DATA_SIZE_PER_TH		(NR_SLOTS * 1)

// Make hash tables OVERHEAD times larger than necessary to store the average
// number of elements per row. The ideal value is as small as possible to
// reduce memory usage, but not too small or else elements are dropped from the
// hash tables.
//
// The actual number of elements per row is closer to the theoretical average
// (less variance) when NR_ROWS_LOG is small. So accordingly OVERHEAD can be
// smaller.
//
// Even (as opposed to odd) values of OVERHEAD sometimes significantly decrease
// performance as they cause VRAM channel conflicts.
#if NR_ROWS_LOG == 16
#error "NR_ROWS_LOG = 16 is currently broken - do not use"
#define OVERHEAD                        3
#elif NR_ROWS_LOG == 18
#define OVERHEAD                        3
#elif NR_ROWS_LOG == 19
#define OVERHEAD                        5
#elif NR_ROWS_LOG == 20 && OPTIM_SIMPLIFY_ROUND
#define OVERHEAD                        6
#elif NR_ROWS_LOG == 20
#define OVERHEAD                        9
#endif

#define NR_ROWS                         (1 << NR_ROWS_LOG)
#define NR_SLOTS            ((1 << (APX_NR_ELMS_LOG - NR_ROWS_LOG)) * OVERHEAD)
// Length of 1 element (slot) in bytes
#define SLOT_LEN                        32
// Total size of hash table
#define HT_SIZE				(NR_ROWS * NR_SLOTS * SLOT_LEN)
// Length of Zcash block header, nonce (part of header)
#define ZCASH_BLOCK_HEADER_LEN		140
// Offset of nTime in header
#define ZCASH_BLOCK_OFFSET_NTIME        (4 + 3 * 32)
// Length of nonce
#define ZCASH_NONCE_LEN			32
// Length of encoded representation of solution size
#define ZCASH_SOLSIZE_LEN		3
// Solution size (1344 = 0x540) represented as a compact integer, in hex
#define ZCASH_SOLSIZE_HEX               "fd4005"
// Length of encoded solution (512 * 21 bits / 8 = 1344 bytes)
#define ZCASH_SOL_LEN                   ((1 << PARAM_K) * (PREFIX + 1) / 8)
// Last N_ZERO_BYTES of nonce must be zero due to my BLAKE2B optimization
#define N_ZERO_BYTES			12
// Number of bytes Zcash needs out of Blake
#define ZCASH_HASH_LEN                  50
// Number of wavefronts per SIMD for the Blake kernel.
// Blake is ALU-bound (beside the atomic counter being incremented) so we need
// at least 2 wavefronts per SIMD to hide the 2-clock latency of integer
// instructions. 10 is the max supported by the hw.
#define BLAKE_WPS               	10
// Maximum number of solutions reported by kernel to host
#define MAX_SOLS			10
// Length of SHA256 target
#define SHA256_TARGET_LEN               (256 / 8)

#if (NR_SLOTS < 16)
#define BITS_PER_ROW 4
#define ROWS_PER_UINT 8
#define ROW_MASK 0x0F
#else
#define BITS_PER_ROW 8
#define ROWS_PER_UINT 4
#define ROW_MASK 0xFF
#endif

// Optional features
#undef ENABLE_DEBUG

/*
** Return the offset of Xi in bytes from the beginning of the slot.
*/
#define xi_offset_for_round(round)	(8 + ((round) / 2) * 4)

// An (uncompressed) solution stores (1 << PARAM_K) 32-bit values
#define SOL_SIZE			((1 << PARAM_K) * 4)

typedef struct sols_s
{
	uint nr;
	uint likely_invalids;
	uchar valid[MAX_SOLS];
	uint values[MAX_SOLS][(1 << PARAM_K)];
} sols_t;
