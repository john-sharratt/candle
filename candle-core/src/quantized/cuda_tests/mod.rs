//! GPU tests and benches for the quantized kernels in candle-core, one file per
//! kernel family.
//!
//! The files gate the int8 matmuls and their activation format, the KO layouts
//! and their producers, the bucketize and CPU-class dispatch for MoE, the live
//! expert launch, the routing kernels, and the quantize round trips. Gates check
//! raw bytes or bit patterns rather than error tolerances wherever the format
//! defines the bytes. The benches are ignored by default and run with `--ignored`.
//!
//! - `tests_bench_gemm`: throughput benches for quantize and int8 matmul kernels.
//! - `tests_bench_moe_sim`: pinned-host and launch-simulation benches for live dispatch.
//! - `tests_bench_probes`: profiler probes, the int8 crossover and read bandwidth.
//! - `tests_bucketize_digest`: pinned digest over every bucketize output byte.
//! - `tests_bucketize_dispatch`: bucketize device-launch gates and grouped matmul equivalence.
//! - `tests_bucketize_live`: live-table bucketize, promotion, read-ahead and victims.
//! - `tests_bucketize_ref`: the CPU reference for bucketize.
//! - `tests_cpu_class`: the CPU expert class split, publication and round trip.
//! - `tests_dense_matmul`: dense tensor-core matmuls at every batch width.
//! - `tests_int8_gemm`: int8 matmul gates against FP16 and CPU references.
//! - `tests_ko_helpers`: host-side KO layout producers used as references.
//! - `tests_ko_submajor`: sub-major KO layouts against independent references.
//! - `tests_ko_twin_repack`: repacking into KO twins on the load path.
//! - `tests_kv_path`: scalar KV-path quantizer byte tests.
//! - `tests_launch_cost_scatter`: launch-cost probes and the scatter's define-not-accumulate contract.
//! - `tests_live_launch`: live expert launch bit-identity with late-published experts.
//! - `tests_q8a128_format`: the q8a128 byte format and its fused producers.
//! - `tests_qkv`: fused q, k and v segmented int8 matmul gates.
//! - `tests_qmatmul_dispatch`: int8 mode flag, pairing guard and MXFP4 kernels.
//! - `tests_quant_roundtrip`: GPU quantize and dequantize round trips against the CPU codec.
//! - `tests_routing`: Sinkhorn, moe_route and look-ahead vote gates.

// Test code: loop indices are block/lane/expert coordinates in flat-buffer
// offset arithmetic, and the tuple returns mirror the kernel's raw parameter
// lists rather than modelling a domain type.
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::manual_div_ceil,
    clippy::needless_question_mark
)]

use super::*;
use candle_kernels::simple::moe_bucketize::{
    run_moe_bucketize, CpuSplit, BUCKETIZE_LAUNCHED, CARRY_FLAG, CPU_CARRY_WORDS, CPU_MAX_N,
    CPU_PROMOTE_ALL, CPU_WORK_SLOT_BYTES, HEADER_WORDS, PROMO_EMPTY, PROMO_SKIP,
    SUMMARY_COUNT_MASK, SUMMARY_CPU_BIT,
};
use cudarc::driver::sys::{cuMemFreeHost, CUstream};
use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use rand::Rng;
use std::ffi::c_void;
use std::time::Instant;

mod tests_bench_gemm;
mod tests_bench_moe_sim;
mod tests_bench_probes;
mod tests_bucketize_digest;
mod tests_bucketize_dispatch;
mod tests_bucketize_live;
mod tests_bucketize_ref;
mod tests_cpu_class;
mod tests_dense_matmul;
mod tests_int8_gemm;
mod tests_ko_helpers;
mod tests_ko_submajor;
mod tests_ko_twin_repack;
mod tests_kv_path;
mod tests_launch_cost_scatter;
mod tests_live_launch;
mod tests_q8a128_format;
mod tests_qkv;
mod tests_qmatmul_dispatch;
mod tests_quant_roundtrip;
mod tests_routing;

use tests_bucketize_live::*;
use tests_bucketize_ref::*;
use tests_cpu_class::*;
use tests_dense_matmul::*;
use tests_int8_gemm::*;
use tests_ko_helpers::*;
use tests_live_launch::*;
use tests_q8a128_format::*;
