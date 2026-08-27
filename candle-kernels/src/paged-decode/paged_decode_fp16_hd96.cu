// FP16 decode, head_dim 96. See `paged_decode_hd_bf16.cuh` for why each head
// dim is compiled as its own translation unit.
#define DECODE_HD 96
#include "paged_decode_hd_fp16.cuh"
