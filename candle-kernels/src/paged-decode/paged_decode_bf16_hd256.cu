// BF16 decode, head_dim 256. See `paged_decode_hd_bf16.cuh` for why each head
// dim is compiled as its own translation unit.
#define DECODE_HD 256
#include "paged_decode_hd_bf16.cuh"
