# Kernel Instantiation System

## Overview

The `impl/` folder contains a modular kernel instantiation system that generates specialized kernels for each combination of:
- **10 quantization formats**: Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0
- **4 Y vector types**: Q8 (quantized), F16 (half precision), BF16 (bfloat16), F8 (fp8)
- **2 kernel variants**: _var (generic) and _var_tc (tensor core optimized)

**Total: 10 × 4 × 2 = 80 kernel variants** in **40 .cu files**

## File Structure

```
impl/
├── kernel_instantiate.cuh      # Macro template for kernel instantiation
├── common.cuh                  # Common infrastructure (included after Y_TYPE)
│
└── [40 kernel files]
    ├── q2_K_q8.cu             # Q2_K quantization + Q8 Y vectors
    ├── q2_K_f16.cu            # Q2_K quantization + F16 Y vectors
    ├── q2_K_bf16.cu           # Q2_K quantization + BF16 Y vectors
    ├── q2_K_f8.cu             # Q2_K quantization + F8 Y vectors
    │
    ├── q3_k_q8.cu
    ├── q3_k_f16.cu
    ├── q3_k_bf16.cu
    ├── q3_k_f8.cu
    │
    ├── [36 more files following same pattern]
    │
    └── q8_0_f8.cu
```

## Include Chain (per .cu file)

Each .cu file follows this include order:

```cuda
// Comments
#define Y_TYPE_Q8              // ← Choose ONE: Q8, F16, BF16, or F8

#include "kernel_instantiate.cuh"  // ← Macro definition
#include "common.cuh"              // ← Infrastructure (after Y_TYPE)
#include "../loader/q2_K.cuh"      // ← Quantization loader (with #ifdef guards)

// Instantiate kernels with macro
INSTANTIATE_KERNEL_PAIR(
    q2_K_q8,
    QK2_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
    vec_dot_q2_K_q8_1,
    block_q8_1, __nv_bfloat16      // ← act_t (activation type), output type
)
```

## Key Components

### 1. `kernel_instantiate.cuh`
Defines the `INSTANTIATE_KERNEL_PAIR` macro that generates both _var and _var_tc kernels:

```cuda
#define INSTANTIATE_KERNEL_PAIR(name, qk, qi, block_type, vdr, vec_dot_fn, act_t, dst_t)
// Generates:
//   - name##_var: Generic kernel (non-tensor-core)
//   - name##_var_tc: Tensor-core optimized variant
```

### 2. `common.cuh`
Provides all common infrastructure needed by kernels:

- **CUDA headers**: `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`
- **Configuration macros**: QK_K, K_SCALE_SIZE, LAUNCH_BOUNDS_*
- **Type definitions**: dfloat, dfloat2, accumulator_type
- **Utility functions**: warp_reduce_sum, to_f32 (type conversions)
- **Kernel function template**: `quantized_gemv` (forward declaration)

### 3. `q{format}_{y_type}.cu` (40 files)
Each instantiation file:
1. Defines Y_TYPE_{type} macro to enable specific Y-vector path in loader
2. Includes infrastructure in correct order
3. Calls INSTANTIATE_KERNEL_PAIR with format-specific parameters

## Y_TYPE Conditional Compilation

Each quantization loader (`../loader/q*.cuh`) wraps Y-vector implementations with `#ifdef Y_TYPE_*` guards:

```cuda
#ifdef Y_TYPE_Q8
    __device__ __forceinline__ float dot_y_q8(const block_q8_1 * y) const { ... }
#endif

#ifdef Y_TYPE_F16
    __device__ __forceinline__ float dot_y_f16(const __half * y) const { ... }
#endif
// ... etc
```

When compiling q2_K_q8.cu:
- `#define Y_TYPE_Q8` enables only Q8 path
- Other paths (F16, BF16, F8) are compiled-out
- Dispatcher chooses correct specialization with constexpr

## Kernel Naming Convention

Each instantiation produces **two external kernels**:

| File | _var Kernel | _var_tc Kernel |
|------|------------|----------------|
| q2_K_q8.cu | `q2_K_q8_var` | `q2_K_q8_var_tc` |
| q4_1_f16.cu | `q4_1_f16_var` | `q4_1_f16_var_tc` |
| q5_k_bf16.cu | `q5_k_bf16_var` | `q5_k_bf16_var_tc` |

## Compilation

All 40 .cu files are compiled independently, each generating:
- 2 kernel function symbols (_var and _var_tc)
- Object files linked into main executable

This provides fine-grained control:
- Disable specific Y_TYPE via build config
- Optimize only needed format+precision combinations
- Reduce binary bloat if certain combinations unused

## Example Usage

To add support for a new quantization format (e.g., Q7_X):

1. Create `../loader/q7_x.cuh` with:
   - `block_q7_x` struct
   - `vec_dot_q_loader_q7_x` class with #ifdef wrapped methods
   - `vec_dot_loader_for<block_q7_x>` specialization

2. Create 4 instantiation files:
   - `q7_x_q8.cu`
   - `q7_x_f16.cu`
   - `q7_x_bf16.cu`
   - `q7_x_f8.cu`

3. Each file follows the standard template - no changes to infrastructure needed.
