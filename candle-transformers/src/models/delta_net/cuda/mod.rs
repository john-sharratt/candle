//! CUDA wrappers for the Gated DeltaNet kernels, checked against the sequential
//! reference.
//!
//! A short span, such as a decode row, a verify block or a short prompt, reads its
//! state and writes a timeline. `delta_net_fold` applies the rows that a step
//! keeps, in place. A long span writes its advanced state to a separate buffer.
//! All tensors are F32 and contiguous, dimensions are validated, and every
//! launcher reports a shape it refuses.
//!
//! - `decode`: the batched decode step and conv.
//! - `fold`: applying kept timeline rows.
//! - `fused`: the wave tensors and their geometry.
//! - `norm_gate`: the output epilogue.
//! - `pointers`: pointer-table layouts and address checks.
//! - `prefill`: the prefill conv, scan and short-span kernels.
//! - `tables`: building pointer and extent tables.
//! - `tests_decode`: decode and conv kernel tests.
//! - `tests_fold`: fold equivalence tests.
//! - `tests_mixed`: mixed-wave tests.
//! - `tests_prefill`: prefill kernel tests.
//! - `tests_support`: test fixtures.

use candle::cuda_backend::cudarc::cublas::sys as cublas;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::quantized::cuda::{produce_q8a128, Q8a128Operand};
use candle::quantized::SumScale;
use candle::{DType, Device, Error, LiveTensor, Result, Storage, Tensor};
pub use candle_kernels::delta_net::DELTA_NET_PREFILL_DIM;
use candle_kernels::delta_net::{
    run_delta_net_batch_ptrs, run_delta_net_conv_decode_f32, run_delta_net_conv_prefill_f32,
    run_delta_net_decode_step_f32, run_delta_net_fold_f32, run_delta_net_norm_gate_f32,
    run_delta_net_prefill_intra_f32, run_delta_net_prefill_state_f32, run_delta_net_short_span_f32,
    DELTA_NET_MAX_SPANS, DELTA_NET_PREFILL_CHUNK, DELTA_NET_PREFILL_CONV_MAX,
};
pub use candle_kernels::delta_net::{DELTA_NET_SHORT_SPAN_CONV, DELTA_NET_SHORT_SPAN_ROWS};

use super::mix::{DeltaNetLayerTable, DeltaNetSeq, DeltaNetSpanTable, DeltaNetState, SeqSpan};
#[cfg(test)]
use super::span_arm::SpanHook;
use super::state_store::RecurrentStateStore;
use super::timeline::{TimelineLayout, TIMELINE_ROWS};
use super::types::DeltaNetDims;
use super::ZGate;
use crate::models::held_tensor::held_empty;
use crate::models::wave_buffers::{upload_into, wave_from_vec_ticketed};
use candle::wave_provenance::WaveTicket;
use std::ffi::c_void;

mod decode;
mod fold;
mod fused;
mod norm_gate;
mod pointers;
mod prefill;
mod tables;
#[cfg(test)]
mod tests_decode;
#[cfg(test)]
mod tests_fold;
#[cfg(test)]
mod tests_mixed;
#[cfg(test)]
mod tests_prefill;
#[cfg(test)]
mod tests_support;

pub use decode::*;
pub use fold::*;
pub use fused::*;
pub use norm_gate::*;
pub use pointers::*;
pub use prefill::*;
pub use tables::*;
