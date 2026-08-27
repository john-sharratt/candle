// BF16 decode, head_dim 64. See `paged_decode_hd_bf16.cuh` for why each head
// dim is compiled as its own translation unit.
#define DECODE_HD 64
#include "paged_decode_hd_bf16.cuh"
