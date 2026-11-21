// =============================================================================
// QUANTIZED OPERATIONS DISPATCHER
// =============================================================================
// Provides extern "C" entry points that dispatch to the appropriate
// quantized kernels based on dtype parameters.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Include block type definitions for fused transpose+quantize kernels
#include "../blocks.cuh"

// QTYPE_* symbolic constants (GgmlDType-aligned numbering, source of truth)
#include "../quantized/block_compact.cuh"

// Include batched fused transpose+quantize kernels
#include "../quantize/transpose_batch.cuh"

// Include palette4 format conversion kernel (new replacement, coexists with above)
#include "../quantize/palette4_convert.cuh"

// Include adaptive per-block format selection kernel
#include "../quantize/select_kv_format.cuh"

// The INT8 decode/prefill element helpers, for the Q0_V decode oracle below.
#include "../convert/int8_elem.cuh"

// q8a128 activation-block quantize/dequant kernel templates (definitions must be
// visible here for the run_quantize_q8a128 / run_dequantize_q8a128 instantiations
// and the unified run_*_block dispatch cases below).
#include "../quantize/quantize_q8a128.cuh"
#include "../dequant/dequant_q8a128.cuh"

// KO weight quantize/dequant kernel templates (F32 ↔ lane-major per-128 KO chunks),
// for the run_quantize_ko / run_dequantize_ko entry points below.
#include "../quantize/quantize_ko.cuh"
#include "../dequant/dequant_ko.cuh"

#include "quantized_dispatcher/forward_decls_and_helpers.cuh"

#include "quantized_dispatcher/run_dispatchers.cuh"

#include "quantized_dispatcher/oracle_and_roundtrip_kernels.cuh"
