//! Samples the next token for a batch of sequences in one dispatch, on the CUDA
//! sampling kernel where a card is present and on the host otherwise.
//!
//! Each sequence's state (its token counts, its recent history and its
//! degenerate-decode run) belongs to the caller, and each row samples on its own
//! dials, so a mixed batch does not carry one row's settings into another. A row
//! is scored against the state as it stands and committed onto that state only
//! when the row key still matches. A speculative block is scored in one launch
//! and rewound afterwards, and a dispatch that fails leaves every state as it
//! found it. The kernel's count tables are stamped for each launch and cleared
//! after it.
//!
//! - `sampler`: the handle that drives the kernel over caller-owned sequences.
//! - `rows`: the inputs the kernel reads for each row, taken from its sequence.
//! - `seq_dials`: the per-row dials, in the layout of the kernel's C struct.
//! - `sequence_state`: per-sequence counts and history that persist across turns.
//! - `kernel`: one launch and one readback over a dispatch's full-vocabulary rows.
//! - `host_sample`: the host sampler's pick for one full-vocabulary row.
//! - `score`: scoring a block of rows without committing any of them.
//! - `commit`: committing a scored row and resolving the token it takes.
//! - `allow_list`: sampling a row constrained to its grammar's allow-list.
//! - `segments`: the segment-close override and the close before a tool call.
//! - `typical`: the typical-acceptance rule over the kernel's distribution.
//! - `tests_allow_list`: tests for the allow-list path.
//! - `tests_batch`: tests for batches mixing sampled, constrained and bad rows.
//! - `tests_degenerate`: tests for the degenerate-decode guard and the EOS failsafes.
//! - `tests_helpers`: fixtures shared by the sampler's unit tests.
//! - `tests_penalties`: tests for the penalties and the per-row dials.
//! - `tests_scoring`: tests for scoring a block of drafts in one dispatch.
//! - `tests_segments`: tests for the segment-close override tiers and state machine.
//! - `tests_stencil`: tests for the per-row allow-list and banned-token masking.
//! - `tests_wave`: tests that every row of a mixed wave uses its own config.

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
