// =============================================================================
// AFFINE OPERATIONS DISPATCHER
// =============================================================================
// Provides a single extern "C" entry point that dispatches to the appropriate
// typed affine kernel based on dtype parameter.
// Affine operation: out = x * mul + add
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// =============================================================================
// Forward declarations of wrapper functions (defined in api.cu)
// These wrappers handle <<<>>> kernel launch configuration internally
// =============================================================================

extern "C" void run_affine_f32(const float* inp, float* out, size_t numel, size_t num_dims, const size_t* info, float mul, float add);
extern "C" void run_affine_f64(const double* inp, double* out, size_t numel, size_t num_dims, const size_t* info, double mul, double add);
extern "C" void run_affine_f16(const void* inp, void* out, size_t numel, size_t num_dims, const size_t* info, float mul, float add);
extern "C" void run_affine_bf16(const void* inp, void* out, size_t numel, size_t num_dims, const size_t* info, float mul, float add);
extern "C" void run_affine_u8(const uint8_t* inp, uint8_t* out, size_t numel, size_t num_dims, const size_t* info, uint8_t mul, uint8_t add);
extern "C" void run_affine_u32(const uint32_t* inp, uint32_t* out, size_t numel, size_t num_dims, const size_t* info, uint32_t mul, uint32_t add);
extern "C" void run_affine_i16(const int16_t* inp, int16_t* out, size_t numel, size_t num_dims, const size_t* info, int16_t mul, int16_t add);
extern "C" void run_affine_i32(const int32_t* inp, int32_t* out, size_t numel, size_t num_dims, const size_t* info, int32_t mul, int32_t add);
extern "C" void run_affine_i64(const int64_t* inp, int64_t* out, size_t numel, size_t num_dims, const size_t* info, int64_t mul, int64_t add);

// =============================================================================
// DType enum values
// =============================================================================
// 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3, 5=u8, 6=u32, 7=i16, 8=i32, 9=i64

// =============================================================================
// Affine dispatcher
// =============================================================================
// Takes mul and add as f64 and converts to appropriate type
extern "C" void run_affine(
    int32_t dtype,
    size_t numel,
    size_t num_dims,
    const size_t* info,
    const void* inp,
    void* out,
    double mul,
    double add
) {
    switch (dtype) {
        case 0: // f32
            run_affine_f32((const float*)inp, (float*)out, numel, num_dims, info, (float)mul, (float)add);
            break;
        case 1: // f64
            run_affine_f64((const double*)inp, (double*)out, numel, num_dims, info, mul, add);
            break;
        case 2: // f16
            run_affine_f16(inp, out, numel, num_dims, info, (float)mul, (float)add);
            break;
        case 3: // bf16
            run_affine_bf16(inp, out, numel, num_dims, info, (float)mul, (float)add);
            break;
        case 4: // f8_e4m3
            // f8_e4m3 not yet supported in wrapper functions
            break;
        case 5: // u8
            run_affine_u8((const uint8_t*)inp, (uint8_t*)out, numel, num_dims, info, (uint8_t)mul, (uint8_t)add);
            break;
        case 6: // u32
            run_affine_u32((const uint32_t*)inp, (uint32_t*)out, numel, num_dims, info, (uint32_t)mul, (uint32_t)add);
            break;
        case 7: // i16
            run_affine_i16((const int16_t*)inp, (int16_t*)out, numel, num_dims, info, (int16_t)mul, (int16_t)add);
            break;
        case 8: // i32
            run_affine_i32((const int32_t*)inp, (int32_t*)out, numel, num_dims, info, (int32_t)mul, (int32_t)add);
            break;
        case 9: // i64
            run_affine_i64((const int64_t*)inp, (int64_t*)out, numel, num_dims, info, (int64_t)mul, (int64_t)add);
            break;
        default:
            // Unknown dtype, do nothing
            break;
    }
}

