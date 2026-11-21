//! Batched sampling wrapper around the CUDA kernel.
//!
//! Provides a high-level Rust API for the batched sampling kernel with:
//! - GPU buffer management
//! - Automatic dtype dispatching
//!
//! State (token counts, recent history) is owned by the caller (DecodeState).

use crate::banned_rows::banned_buffer;
use crate::config::SamplingConfig;
use crate::line_ends::ends_a_line;
use crate::penalty_counts::{count_table_ptr, PenaltyTables, SparseCounts};
use crate::sampler_args::{refuse_nonfinite, split_readback, ArgPack};
use crate::scheduler::profile;
use crate::stencil::ban;
use crate::token_buffer::TokenBuffer;
use candle::cuda_backend::{CudaStorageSlice, Held};
use candle::{DType, Device, IndexOp, Tensor};
use candle_kernels::sampling::{run_batched_sampling, DType as KernelDType, ALLOW_NOT_FINITE};
use candle_transformers::generation::{LogitsProcessor, PendingSample};
use candle_transformers::models::speculative_choice::TypicalAcceptance;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use std::sync::Mutex;

mod allow_list;
mod commit;
mod host_sample;
mod kernel;
mod rows;
mod sampler;
mod score;
mod segments;
mod seq_dials;
mod sequence_state;
#[cfg(test)]
mod tests_allow_list;
#[cfg(test)]
mod tests_batch;
#[cfg(test)]
mod tests_degenerate;
#[cfg(test)]
mod tests_helpers;
#[cfg(test)]
mod tests_penalties;
#[cfg(test)]
mod tests_scoring;
#[cfg(test)]
mod tests_segments;
#[cfg(test)]
mod tests_stencil;
#[cfg(test)]
mod tests_wave;
mod typical;

use allow_list::*;
use host_sample::*;
pub use rows::*;
pub use sampler::*;
use segments::*;
use seq_dials::*;
pub use sequence_state::*;
use typical::*;
