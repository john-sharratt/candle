//! Paged attention over chunked KV arenas: the decode and prefill kernels that
//! read each slot through its header, and the staging and launch-shape logic they
//! share.
//!
//! Each kernel reads a slot's chunks and position map directly, so no copy of the
//! cache is made for the attention. Staging and the launch cut decide how the
//! prefill work is divided, and the output projection consumes the result in the
//! form the kernel wrote it.
//!
//! - `decode_attn`: paged decode attention over the slot headers.
//! - `glue_attn`: the reprojection glue forward.
//! - `int8_prefill`: the INT8 prefix-attention prefill launch.
//! - `kv_stage`: pre-staged keys and values for bulk prefill sequences.
//! - `paged_prefill`: prefill over chunked arenas.
//! - `slot_headers`: the per-slot headers and the layer-invariant position map.
//! - Tests: `tests_correctness`, `tests_launch`, `tests_pm_divergence`,
//!   `tests_support`.

#[cfg(feature = "cuda")]
use crate::models::profile::{gpu_span, pipeline_record, pipeline_record_duration, profile_now};
use candle::quantized::pinned_staging::Generation;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::GpuBuf;
use candle::*;
pub(crate) use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
pub(crate) use candle_nn::kv_cache::CHUNK_SIZE;

#[cfg(feature = "cuda")]
use {
    candle::backend::BackendStorage,
    candle::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr},
    candle_kernels::paged_glue::{run_paged_glue_bf16, run_paged_glue_fp16},
    candle_kernels::paged_prefill::*,
    candle_nn::kv_cache::{
        prefill_kv_stage_bytes, prefill_kv_stage_positions, ArenaFormatTag, ChunkedKvBacking,
        PrefillKvStageLayout, PrefillLaunchBounds, BUMP_ALIGNMENT, PREFILL_KV_STAGE_MIN_Q_LEN,
    },
    core::ffi::c_void,
    half::{bf16, f16},
};

#[cfg(feature = "cuda")]
use crate::models::operand_guard::expect_dtype;
use crate::models::qsa_selection::QsaSelection;
use crate::models::rope_schedule::RopeRungs;
#[cfg(feature = "cuda")]
use crate::models::slot_header::{SlotHeaderHost, SLOT_HEADER_BYTES};
#[cfg(feature = "cuda")]
use crate::models::slot_state::{tensor_u8_device_ptr, SlotTokenLayout};
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_from_vec_ticketed;
#[cfg(feature = "cuda")]
use candle_kernels::rope::RopeRungsFfi;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::ChunkPin;
#[cfg(feature = "cuda")]
use std::cell::RefCell;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::models::wave_buffers::KernelOutput;
#[cfg(feature = "cuda")]
use candle::engine_events::note;
#[cfg(feature = "cuda")]
use candle::LiveTensor;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::WaveGeneration;

mod decode_attn;
mod glue_attn;
mod int8_prefill;
mod kv_stage;
mod paged_prefill;
mod slot_headers;
#[cfg(test)]
mod tests_correctness;
#[cfg(test)]
mod tests_launch;
#[cfg(test)]
mod tests_pm_divergence;
#[cfg(test)]
mod tests_support;

pub use decode_attn::*;
pub use glue_attn::*;
pub(crate) use int8_prefill::*;
pub use kv_stage::*;
pub use paged_prefill::*;
pub use slot_headers::*;
