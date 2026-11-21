// `extern "C"` wrappers for dtype casts, the indexing families, the additional
// where kernels, fused SiLU-mul and the MoE gather, route, vote and scatter
// kernels.
//
// The cast wrappers between every pair of dtypes, then the `is_`, `gather_`,
// `ia_`, `sa_` and `s_` indexing wrappers by index and value type, the
// remaining `where_` wrappers, and the fused SiLU-mul and MoE kernels.
// These are the wrappers the MoE path and the indexing ops call directly, so
// they sit together.

// --- Cast kernels (additional) ---
extern "C" __global__ void cast_f32_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void cast_f64_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void cast_u8_u8(const size_t, const size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void cast_u32_u32(const size_t, const size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void cast_i64_i64(const size_t, const size_t, const size_t*, const int64_t*, int64_t*);
extern "C" __global__ void cast_f64_f32(const size_t, const size_t, const size_t*, const double*, float*);
extern "C" __global__ void cast_f32_f64(const size_t, const size_t, const size_t*, const float*, double*);
extern "C" __global__ void cast_f32_u8(const size_t, const size_t, const size_t*, const float*, uint8_t*);
extern "C" __global__ void cast_u8_f32(const size_t, const size_t, const size_t*, const uint8_t*, float*);
extern "C" __global__ void cast_f32_u32(const size_t, const size_t, const size_t*, const float*, uint32_t*);
extern "C" __global__ void cast_u32_f32(const size_t, const size_t, const size_t*, const uint32_t*, float*);
extern "C" __global__ void cast_f32_i64(const size_t, const size_t, const size_t*, const float*, int64_t*);
extern "C" __global__ void cast_i64_f32(const size_t, const size_t, const size_t*, const int64_t*, float*);
extern "C" __global__ void cast_f64_u8(const size_t, const size_t, const size_t*, const double*, uint8_t*);
extern "C" __global__ void cast_u8_f64(const size_t, const size_t, const size_t*, const uint8_t*, double*);
extern "C" __global__ void cast_f64_u32(const size_t, const size_t, const size_t*, const double*, uint32_t*);
extern "C" __global__ void cast_u32_f64(const size_t, const size_t, const size_t*, const uint32_t*, double*);
extern "C" __global__ void cast_f64_i64(const size_t, const size_t, const size_t*, const double*, int64_t*);
extern "C" __global__ void cast_i64_f64(const size_t, const size_t, const size_t*, const int64_t*, double*);
extern "C" __global__ void cast_u8_u32(const size_t, const size_t, const size_t*, const uint8_t*, uint32_t*);
extern "C" __global__ void cast_u32_u8(const size_t, const size_t, const size_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void cast_u8_i64(const size_t, const size_t, const size_t*, const uint8_t*, int64_t*);
extern "C" __global__ void cast_i64_u8(const size_t, const size_t, const size_t*, const int64_t*, uint8_t*);
extern "C" __global__ void cast_u32_i64(const size_t, const size_t, const size_t*, const uint32_t*, int64_t*);
extern "C" __global__ void cast_i64_u32(const size_t, const size_t, const size_t*, const int64_t*, uint32_t*);
extern "C" __global__ void cast_f16_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void cast_bf16_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void cast_f64_f16(const size_t, const size_t, const size_t*, const double*, __half*);
extern "C" __global__ void cast_f16_f64(const size_t, const size_t, const size_t*, const __half*, double*);
extern "C" __global__ void cast_f64_bf16(const size_t, const size_t, const size_t*, const double*, __nv_bfloat16*);
extern "C" __global__ void cast_bf16_f64(const size_t, const size_t, const size_t*, const __nv_bfloat16*, double*);
extern "C" __global__ void cast_f16_bf16(const size_t, const size_t, const size_t*, const __half*, __nv_bfloat16*);
extern "C" __global__ void cast_bf16_f16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __half*);
extern "C" __global__ void cast_f16_u8(const size_t, const size_t, const size_t*, const __half*, uint8_t*);
extern "C" __global__ void cast_u8_f16(const size_t, const size_t, const size_t*, const uint8_t*, __half*);
extern "C" __global__ void cast_f16_u32(const size_t, const size_t, const size_t*, const __half*, uint32_t*);
extern "C" __global__ void cast_u32_f16(const size_t, const size_t, const size_t*, const uint32_t*, __half*);
extern "C" __global__ void cast_bf16_u8(const size_t, const size_t, const size_t*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void cast_u8_bf16(const size_t, const size_t, const size_t*, const uint8_t*, __nv_bfloat16*);
extern "C" __global__ void cast_bf16_u32(const size_t, const size_t, const size_t*, const __nv_bfloat16*, uint32_t*);
extern "C" __global__ void cast_u32_bf16(const size_t, const size_t, const size_t*, const uint32_t*, __nv_bfloat16*);

// --- Additional indexing kernels ---
extern "C" __global__ void is_i64_f32(const size_t, const size_t, const size_t*, const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_f64(const size_t, const size_t, const size_t*, const int64_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_f16(const size_t, const size_t, const size_t*, const int64_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_bf16(const size_t, const size_t, const size_t*, const int64_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_u8(const size_t, const size_t, const size_t*, const int64_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_u32(const size_t, const size_t, const size_t*, const int64_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_f32(const size_t, const size_t, const size_t*, const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_f64(const size_t, const size_t, const size_t*, const uint32_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_f16(const size_t, const size_t, const size_t*, const uint32_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_bf16(const size_t, const size_t, const size_t*, const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_u8(const size_t, const size_t, const size_t*, const uint32_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_i64(const size_t, const size_t, const size_t*, const uint32_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_f32(const size_t, const size_t, const size_t*, const uint8_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_f64(const size_t, const size_t, const size_t*, const uint8_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_f16(const size_t, const size_t, const size_t*, const uint8_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_bf16(const size_t, const size_t, const size_t*, const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_u32(const size_t, const size_t, const size_t*, const uint8_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_i64(const size_t, const size_t, const size_t*, const uint8_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void gather_i64_f32(const size_t, const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_f64(const size_t, const int64_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_f16(const size_t, const int64_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_bf16(const size_t, const int64_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_u8(const size_t, const int64_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_u32(const size_t, const int64_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_i64(const size_t, const int64_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_f32(const size_t, const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_f64(const size_t, const uint32_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_f16(const size_t, const uint32_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_bf16(const size_t, const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_u8(const size_t, const uint32_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_u32(const size_t, const uint32_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_i64(const size_t, const uint32_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_f32(const size_t, const uint8_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_f64(const size_t, const uint8_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_f16(const size_t, const uint8_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_bf16(const size_t, const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_u8(const size_t, const uint8_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_u32(const size_t, const uint8_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_i64(const size_t, const uint8_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void ia_i64_f32(const int64_t*, const size_t, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_f64(const int64_t*, const size_t, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_f16(const int64_t*, const size_t, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_bf16(const int64_t*, const size_t, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_f64(const uint32_t*, const size_t, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_f16(const uint32_t*, const size_t, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_bf16(const uint32_t*, const size_t, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void sa_i64_f32(const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_f64(const int64_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_f16(const int64_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_bf16(const int64_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_f64(const uint32_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_f16(const uint32_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_bf16(const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void ia_i64_u8(const int64_t*, const size_t, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_u32(const int64_t*, const size_t, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_i64(const int64_t*, const size_t, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_u8(const uint32_t*, const size_t, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_u32(const uint32_t*, const size_t, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_i64(const uint32_t*, const size_t, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_f32(const uint8_t*, const size_t, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_f64(const uint8_t*, const size_t, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_u8(const uint8_t*, const size_t, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_u32(const uint8_t*, const size_t, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_i64(const uint8_t*, const size_t, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_f16(const uint8_t*, const size_t, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_bf16(const uint8_t*, const size_t, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void sa_i64_u8(const int64_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_u32(const int64_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_i64(const int64_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_u8(const uint32_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_u32(const uint32_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_i64(const uint32_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_f32(const uint8_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_f64(const uint8_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_u8(const uint8_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_u32(const uint8_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_i64(const uint8_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_f16(const uint8_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_bf16(const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void ia_i64_f8_e4m3(const int64_t*, const size_t, const __nv_fp8_e4m3*, __nv_fp8_e4m3*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_f8_e4m3(const uint32_t*, const size_t, const __nv_fp8_e4m3*, __nv_fp8_e4m3*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u8_f8_e4m3(const uint8_t*, const size_t, const __nv_fp8_e4m3*, __nv_fp8_e4m3*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_f8_e4m3(const int64_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_f8_e4m3(const uint32_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u8_f8_e4m3(const uint8_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void s_i64_f32(const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_i64_f64(const int64_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_i64_u8(const int64_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_i64_u32(const int64_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_i64_i64(const int64_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_i64_f16(const int64_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_i64_bf16(const int64_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_f32(const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_f64(const uint32_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_u8(const uint32_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_u32(const uint32_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_i64(const uint32_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_f16(const uint32_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_bf16(const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u8_f32(const uint8_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u8_f64(const uint8_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u8_u8(const uint8_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u8_u32(const uint8_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u8_i64(const uint8_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u8_f16(const uint8_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u8_bf16(const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);

// --- Additional where kernels ---
extern "C" __global__ void where_u8_f64(const size_t, const size_t, const size_t*, const uint8_t*, const double*, const double*, double*);
extern "C" __global__ void where_u8_f16(const size_t, const size_t, const size_t*, const uint8_t*, const __half*, const __half*, __half*);
extern "C" __global__ void where_u8_bf16(const size_t, const size_t, const size_t*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void where_u8_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void where_u8_u32(const size_t, const size_t, const size_t*, const uint8_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void where_u8_i64(const size_t, const size_t, const size_t*, const uint8_t*, const int64_t*, const int64_t*, int64_t*);

// --- Fused SiLU-Mul kernel forward declarations ---
extern "C" __global__ void fused_silu_mul_bf16(const unsigned int, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void fused_silu_mul_f16(const unsigned int, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void fused_silu_mul_f32(const unsigned int, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void fused_silu_mul_f8_e4m3(const unsigned int, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
// Vectorized variants
extern "C" __global__ void fused_silu_mul_bf16_vec2(const unsigned int, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void fused_silu_mul_f16_vec2(const unsigned int, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void fused_silu_mul_f32_vec4(const unsigned int, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void fused_silu_mul_f8_e4m3_vec4(const unsigned int, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);

// --- Fused MoE gather/scatter kernel forward declarations ---
extern "C" __global__ void moe_gather_bf16(__nv_bfloat16*, const __nv_bfloat16*, const uint32_t*, size_t, size_t);
extern "C" __global__ void moe_gather_f16(__half*, const __half*, const uint32_t*, size_t, size_t);
extern "C" __global__ void moe_gather_f32(float*, const float*, const uint32_t*, size_t, size_t);
extern "C" __global__ void moe_gather_q8a128_tiles(uint8_t*, const uint8_t*, const uint32_t*, size_t, size_t);
// Fused router: softmax + top-k select + (optional) renormalize, one thread per token.
// The `_x512` variants are the 16-slot instantiation (up to 512 experts).
extern "C" __global__ void moe_route_f32(const float*, uint32_t*, float*, int, int, int, int, int);
extern "C" __global__ void moe_route_f16(const __half*, uint32_t*, float*, int, int, int, int, int);
extern "C" __global__ void moe_route_bf16(const __nv_bfloat16*, uint32_t*, float*, int, int, int, int, int);
extern "C" __global__ void moe_route_f32_x512(const float*, uint32_t*, float*, int, int, int, int, int);
extern "C" __global__ void moe_route_f16_x512(const __half*, uint32_t*, float*, int, int, int, int, int);
extern "C" __global__ void moe_route_bf16_x512(const __nv_bfloat16*, uint32_t*, float*, int, int, int, int, int);
// Router look-ahead: per-hop top-k votes of later layers' routers.
extern "C" __global__ void moe_predict_votes_f32(const float*, uint32_t*, float*, int, int, int, int, int, int);
// Deterministic scatter (sequential per-token reduce, no atomicAdd, variable k via prefix sum)
extern "C" __global__ void deterministic_scatter_bf16(__nv_bfloat16*, const __nv_bfloat16*, const uint32_t*, const float*, const uint32_t*, const int*, int, int);
extern "C" __global__ void deterministic_scatter_f16(__half*, const __half*, const uint32_t*, const float*, const uint32_t*, const int*, int, int);
extern "C" __global__ void deterministic_scatter_f32(float*, const float*, const uint32_t*, const float*, const uint32_t*, const int*, int, int);
