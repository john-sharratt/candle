#include "binary_op_macros.cuh"
#include<stdint.h>

BINARY_OP(__nv_bfloat16, badd_bf16, x + y)
BINARY_OP(__nv_bfloat16, bdiv_bf16, x / y)
BINARY_OP(__nv_bfloat16, bmul_bf16, x * y)
BINARY_OP(__nv_bfloat16, bsub_bf16, x - y)
BINARY_OP(__nv_bfloat16, bmaximum_bf16, maxg(x, y))
BINARY_OP(__nv_bfloat16, bminimum_bf16, ming(x, y))
BINARY_OP_OUT(__nv_bfloat16, uint8_t, eq_bf16, x == y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, ne_bf16, x != y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, lt_bf16, x < y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, le_bf16, x <= y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, gt_bf16, x > y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, ge_bf16, x >= y)

// Vectorized bf16 ops with native bf162 intrinsics
BINARY_OP_BF16_VEC2(badd_bf16, __hadd2, bf_add)
BINARY_OP_BF16_VEC2(bsub_bf16, __hsub2, bf_sub)
BINARY_OP_BF16_VEC2(bmul_bf16, __hmul2, bf_mul)
BINARY_OP_BF16_VEC2(bdiv_bf16, __h2div, bf_div)
BINARY_OP_BF16_VEC2(bmaximum_bf16, __hmax2, bf_max)
BINARY_OP_BF16_VEC2(bminimum_bf16, __hmin2, bf_min)

#define F8E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3))

// FP8 doesn't support __ldg, so use the NO_LDG variants
BINARY_OP_NO_LDG(__nv_fp8_e4m3, badd_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) + F8E4M3_TO_FLOAT(y)))
BINARY_OP_NO_LDG(__nv_fp8_e4m3, bdiv_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) / F8E4M3_TO_FLOAT(y)))
BINARY_OP_NO_LDG(__nv_fp8_e4m3, bmul_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) * F8E4M3_TO_FLOAT(y)))
BINARY_OP_NO_LDG(__nv_fp8_e4m3, bsub_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) - F8E4M3_TO_FLOAT(y)))
BINARY_OP_NO_LDG(__nv_fp8_e4m3, bmaximum_f8_e4m3, maxg(x, y))
BINARY_OP_NO_LDG(__nv_fp8_e4m3, bminimum_f8_e4m3, ming(x, y))
BINARY_OP_OUT_NO_LDG(__nv_fp8_e4m3, uint8_t, eq_f8_e4m3, F8E4M3_TO_FLOAT(x) == F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT_NO_LDG(__nv_fp8_e4m3, uint8_t, ne_f8_e4m3, F8E4M3_TO_FLOAT(x) != F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT_NO_LDG(__nv_fp8_e4m3, uint8_t, lt_f8_e4m3, F8E4M3_TO_FLOAT(x) < F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT_NO_LDG(__nv_fp8_e4m3, uint8_t, le_f8_e4m3, F8E4M3_TO_FLOAT(x) <= F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT_NO_LDG(__nv_fp8_e4m3, uint8_t, gt_f8_e4m3, F8E4M3_TO_FLOAT(x) > F8E4M3_TO_FLOAT(y))
BINARY_OP_OUT_NO_LDG(__nv_fp8_e4m3, uint8_t, ge_f8_e4m3, F8E4M3_TO_FLOAT(x) >= F8E4M3_TO_FLOAT(y))

// Vectorized fp8 ops (4 x fp8 = 32-bit loads)
BINARY_OP_F8E4M3_VEC4(badd_f8_e4m3, f8_add)
BINARY_OP_F8E4M3_VEC4(bsub_f8_e4m3, f8_sub)
BINARY_OP_F8E4M3_VEC4(bmul_f8_e4m3, f8_mul)
BINARY_OP_F8E4M3_VEC4(bdiv_f8_e4m3, f8_div)
BINARY_OP_F8E4M3_VEC4(bmaximum_f8_e4m3, f8_max)
BINARY_OP_F8E4M3_VEC4(bminimum_f8_e4m3, f8_min)

BINARY_OP(__half, badd_f16, x + y)
BINARY_OP(__half, bdiv_f16, x / y)
BINARY_OP(__half, bmul_f16, x * y)
BINARY_OP(__half, bsub_f16, x - y)
BINARY_OP(__half, bmaximum_f16, maxg(x, y))
BINARY_OP(__half, bminimum_f16, ming(x, y))
BINARY_OP_OUT(__half, uint8_t, eq_f16, x == y)
BINARY_OP_OUT(__half, uint8_t, ne_f16, x != y)
BINARY_OP_OUT(__half, uint8_t, lt_f16, x < y)
BINARY_OP_OUT(__half, uint8_t, le_f16, x <= y)
BINARY_OP_OUT(__half, uint8_t, gt_f16, x > y)
BINARY_OP_OUT(__half, uint8_t, ge_f16, x >= y)

// Vectorized fp16 ops with native half2 intrinsics
BINARY_OP_F16_VEC2(badd_f16, __hadd2, h_add)
BINARY_OP_F16_VEC2(bsub_f16, __hsub2, h_sub)
BINARY_OP_F16_VEC2(bmul_f16, __hmul2, h_mul)
BINARY_OP_F16_VEC2(bdiv_f16, __h2div, h_div)
BINARY_OP_F16_VEC2(bmaximum_f16, __hmax2, h_max)
BINARY_OP_F16_VEC2(bminimum_f16, __hmin2, h_min)

// Standard scalar float32 ops
BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_f64, x + y);
BINARY_OP(uint8_t, badd_u8, x + y);
BINARY_OP(uint32_t, badd_u32, x + y);
BINARY_OP(int64_t, badd_i64, x + y);
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y);
BINARY_OP(uint8_t, bdiv_u8, x / y);
BINARY_OP(uint32_t, bdiv_u32, x / y);
BINARY_OP(int64_t, bdiv_i64, x / y);
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y);
BINARY_OP(uint8_t, bmul_u8, x * y);
BINARY_OP(uint32_t, bmul_u32, x * y);
BINARY_OP(int64_t, bmul_i64, x * y);
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
BINARY_OP(uint8_t, bsub_u8, x - y);
BINARY_OP(uint32_t, bsub_u32, x - y);
BINARY_OP(int64_t, bsub_i64, x - y);
BINARY_OP(float, bminimum_f32, ming(x, y));
BINARY_OP(double, bminimum_f64, ming(x, y));
BINARY_OP(uint8_t, bminimum_u8, ming(x, y));
BINARY_OP(uint32_t, bminimum_u32, ming(x, y));
BINARY_OP(int64_t, bminimum_i64, ming(x, y));
BINARY_OP(float, bmaximum_f32, maxg(x, y));
BINARY_OP(double, bmaximum_f64, maxg(x, y));
BINARY_OP(uint8_t, bmaximum_u8, maxg(x, y));
BINARY_OP(uint32_t, bmaximum_u32, maxg(x, y));
BINARY_OP(int64_t, bmaximum_i64, maxg(x, y));

// Vectorized float32 ops (float4 = 128-bit loads)
BINARY_OP_F32_VEC4(badd_f32, +)
BINARY_OP_F32_VEC4(bsub_f32, -)
BINARY_OP_F32_VEC4(bmul_f32, *)
BINARY_OP_F32_DIV_VEC4(bdiv_f32)  // Uses __fdividef
BINARY_OP_F32_MINMAX_VEC4(bminimum_f32, fminf)
BINARY_OP_F32_MINMAX_VEC4(bmaximum_f32, fmaxf)

// Vectorized float64 ops (double2 = 128-bit loads)
BINARY_OP_F64_VEC2(badd_f64, +)
BINARY_OP_F64_VEC2(bsub_f64, -)
BINARY_OP_F64_VEC2(bmul_f64, *)
BINARY_OP_F64_VEC2(bdiv_f64, /)

BINARY_OP_OUT(float, uint8_t, eq_f32, x == y)
BINARY_OP_OUT(double, uint8_t, eq_f64, x == y)
BINARY_OP_OUT(uint8_t, uint8_t, eq_u8, x == y)
BINARY_OP_OUT(uint32_t, uint8_t, eq_u32, x == y)
BINARY_OP_OUT(int64_t, uint8_t, eq_i64, x == y)

BINARY_OP_OUT(float, uint8_t, ne_f32, x != y)
BINARY_OP_OUT(double, uint8_t, ne_f64, x != y)
BINARY_OP_OUT(uint8_t, uint8_t, ne_u8, x != y)
BINARY_OP_OUT(uint32_t, uint8_t, ne_u32, x != y)
BINARY_OP_OUT(int64_t, uint8_t, ne_i64, x != y)

BINARY_OP_OUT(float, uint8_t, lt_f32, x < y)
BINARY_OP_OUT(double, uint8_t, lt_f64, x < y)
BINARY_OP_OUT(uint8_t, uint8_t, lt_u8, x < y)
BINARY_OP_OUT(uint32_t, uint8_t, lt_u32, x < y)
BINARY_OP_OUT(int64_t, uint8_t, lt_i64, x < y)

BINARY_OP_OUT(float, uint8_t, le_f32, x <= y)
BINARY_OP_OUT(double, uint8_t, le_f64, x <= y)
BINARY_OP_OUT(uint8_t, uint8_t, le_u8, x <= y)
BINARY_OP_OUT(uint32_t, uint8_t, le_u32, x <= y)
BINARY_OP_OUT(int64_t, uint8_t, le_i64, x <= y)

BINARY_OP_OUT(float, uint8_t, gt_f32, x > y)
BINARY_OP_OUT(double, uint8_t, gt_f64, x > y)
BINARY_OP_OUT(uint8_t, uint8_t, gt_u8, x > y)
BINARY_OP_OUT(uint32_t, uint8_t, gt_u32, x > y)
BINARY_OP_OUT(int64_t, uint8_t, gt_i64, x > y)

BINARY_OP_OUT(float, uint8_t, ge_f32, x >= y)
BINARY_OP_OUT(double, uint8_t, ge_f64, x >= y)
BINARY_OP_OUT(uint8_t, uint8_t, ge_u8, x >= y)
BINARY_OP_OUT(uint32_t, uint8_t, ge_u32, x >= y)
BINARY_OP_OUT(int64_t, uint8_t, ge_i64, x >= y)

// =============================================================================
// IN-PLACE BINARY OPERATIONS
// =============================================================================

// Scalar helper macros for in-place ops
#define IP_ADD(a, b) ((a) + (b))
#define IP_SUB(a, b) ((a) - (b))
#define IP_MUL(a, b) ((a) * (b))
#define IP_DIV(a, b) ((a) / (b))

// f32 in-place ops (scalar)
BINARY_OP_INPLACE(float, badd_f32_inplace, x + y)
BINARY_OP_INPLACE(float, bsub_f32_inplace, x - y)
BINARY_OP_INPLACE(float, bmul_f32_inplace, x * y)
BINARY_OP_INPLACE(float, bdiv_f32_inplace, x / y)
BINARY_OP_INPLACE(float, bmin_f32_inplace, ming(x, y))
BINARY_OP_INPLACE(float, bmax_f32_inplace, maxg(x, y))

// f32 in-place ops (vectorized)
BINARY_OP_INPLACE_F32_VEC4(badd_f32_inplace, IP_ADD)
BINARY_OP_INPLACE_F32_VEC4(bsub_f32_inplace, IP_SUB)
BINARY_OP_INPLACE_F32_VEC4(bmul_f32_inplace, IP_MUL)
BINARY_OP_INPLACE_F32_VEC4(bdiv_f32_inplace, IP_DIV)

// f64 in-place ops
BINARY_OP_INPLACE(double, badd_f64_inplace, x + y)
BINARY_OP_INPLACE(double, bsub_f64_inplace, x - y)
BINARY_OP_INPLACE(double, bmul_f64_inplace, x * y)
BINARY_OP_INPLACE(double, bdiv_f64_inplace, x / y)
BINARY_OP_INPLACE(double, bmin_f64_inplace, ming(x, y))
BINARY_OP_INPLACE(double, bmax_f64_inplace, maxg(x, y))

// Integer in-place ops
BINARY_OP_INPLACE(uint8_t, badd_u8_inplace, x + y)
BINARY_OP_INPLACE(uint8_t, bsub_u8_inplace, x - y)
BINARY_OP_INPLACE(uint8_t, bmul_u8_inplace, x * y)
BINARY_OP_INPLACE(uint8_t, bdiv_u8_inplace, x / y)
BINARY_OP_INPLACE(uint32_t, badd_u32_inplace, x + y)
BINARY_OP_INPLACE(uint32_t, bsub_u32_inplace, x - y)
BINARY_OP_INPLACE(uint32_t, bmul_u32_inplace, x * y)
BINARY_OP_INPLACE(uint32_t, bdiv_u32_inplace, x / y)
BINARY_OP_INPLACE(int64_t, badd_i64_inplace, x + y)
BINARY_OP_INPLACE(int64_t, bsub_i64_inplace, x - y)
BINARY_OP_INPLACE(int64_t, bmul_i64_inplace, x * y)
BINARY_OP_INPLACE(int64_t, bdiv_i64_inplace, x / y)

// f16 in-place ops (scalar)
BINARY_OP_INPLACE(__half, badd_f16_inplace, x + y)
BINARY_OP_INPLACE(__half, bsub_f16_inplace, x - y)
BINARY_OP_INPLACE(__half, bmul_f16_inplace, x * y)
BINARY_OP_INPLACE(__half, bdiv_f16_inplace, x / y)
BINARY_OP_INPLACE(__half, bmin_f16_inplace, ming(x, y))
BINARY_OP_INPLACE(__half, bmax_f16_inplace, maxg(x, y))

// f16 in-place ops (vectorized)
BINARY_OP_INPLACE_F16_VEC2(badd_f16_inplace, __hadd2, h_add)
BINARY_OP_INPLACE_F16_VEC2(bsub_f16_inplace, __hsub2, h_sub)
BINARY_OP_INPLACE_F16_VEC2(bmul_f16_inplace, __hmul2, h_mul)
BINARY_OP_INPLACE_F16_VEC2(bdiv_f16_inplace, __h2div, h_div)

// bf16 in-place ops (scalar)
BINARY_OP_INPLACE(__nv_bfloat16, badd_bf16_inplace, x + y)
BINARY_OP_INPLACE(__nv_bfloat16, bsub_bf16_inplace, x - y)
BINARY_OP_INPLACE(__nv_bfloat16, bmul_bf16_inplace, x * y)
BINARY_OP_INPLACE(__nv_bfloat16, bdiv_bf16_inplace, x / y)
BINARY_OP_INPLACE(__nv_bfloat16, bmin_bf16_inplace, ming(x, y))
BINARY_OP_INPLACE(__nv_bfloat16, bmax_bf16_inplace, maxg(x, y))

// bf16 in-place ops (vectorized)
BINARY_OP_INPLACE_BF16_VEC2(badd_bf16_inplace, __hadd2, bf_add)
BINARY_OP_INPLACE_BF16_VEC2(bsub_bf16_inplace, __hsub2, bf_sub)
BINARY_OP_INPLACE_BF16_VEC2(bmul_bf16_inplace, __hmul2, bf_mul)
BINARY_OP_INPLACE_BF16_VEC2(bdiv_bf16_inplace, __h2div, bf_div)

// f8_e4m3 in-place ops (scalar using existing F8E4M3_TO_FLOAT)
BINARY_OP_INPLACE(__nv_fp8_e4m3, badd_f8_e4m3_inplace, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) + F8E4M3_TO_FLOAT(y)))
BINARY_OP_INPLACE(__nv_fp8_e4m3, bsub_f8_e4m3_inplace, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) - F8E4M3_TO_FLOAT(y)))
BINARY_OP_INPLACE(__nv_fp8_e4m3, bmul_f8_e4m3_inplace, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) * F8E4M3_TO_FLOAT(y)))
BINARY_OP_INPLACE(__nv_fp8_e4m3, bdiv_f8_e4m3_inplace, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) / F8E4M3_TO_FLOAT(y)))
BINARY_OP_INPLACE(__nv_fp8_e4m3, bmin_f8_e4m3_inplace, ming(x, y))
BINARY_OP_INPLACE(__nv_fp8_e4m3, bmax_f8_e4m3_inplace, maxg(x, y))

// f8_e4m3 in-place ops (vectorized)
BINARY_OP_INPLACE_F8E4M3_VEC4(badd_f8_e4m3_inplace, f8_add)
BINARY_OP_INPLACE_F8E4M3_VEC4(bsub_f8_e4m3_inplace, f8_sub)
BINARY_OP_INPLACE_F8E4M3_VEC4(bmul_f8_e4m3_inplace, f8_mul)
BINARY_OP_INPLACE_F8E4M3_VEC4(bdiv_f8_e4m3_inplace, f8_div)
