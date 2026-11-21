//! Rust wrapper and standalone harness for the paged latent decode and prefill
//! kernels, a single-latent hybrid attention with K and V held as one tensor and
//! a head dimension of 512.
//!
//! Callers drive the kernels through candle tensors. The harness builds a
//! synthetic window slot and gallery with no substrate or model behind them.
//! The kernels are checked against a CPU mirror of their exact arithmetic,
//! which is bit-exact except at a fast-math seam, and against a plain float
//! sink-softmax reference at int8 tolerance. Geometry is verified at load, so a
//! disagreement between host and kernel is refused rather than producing wrong
//! attention.
//!
//! - `band_codec`: host codecs for every band format.
//! - `corpus_cache`: the derived int8 corpus cache.
//! - `decode`: the decode launch.
//! - `glue`: glue scatter into the reserved gaps.
//! - `launch`: the split policy and the shared workspace.
//! - `paged_geometry`: the geometry declaration and its load check.
//! - `prefill`: the prefill launch.
//! - `synthetic`: synthetic slots and the CPU mirror.
//! - Tests: `tests_formats`, `tests_geometry`, `tests_kernel_gates`,
//!   `tests_oracle`, `tests_prefill`.

#![allow(clippy::too_many_arguments)]

use candle::quantized::k_quants;
use candle::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::Generation;
#[cfg(feature = "cuda")]
use candle_kernels::paged_latent::GLUE_SCATTER_WORDS;

#[cfg(feature = "cuda")]
use crate::models::slot_header::{SlotHeaderHost, SLOT_HEADER_BYTES};

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
use super::config::Config;
use super::desc;
use super::geometry::{LatentGeometry, D512_R64_B16};

mod band_codec;
mod corpus_cache;
mod decode;
mod glue;
mod launch;
mod paged_geometry;
mod prefill;
mod synthetic;
#[cfg(test)]
mod tests_formats;
#[cfg(test)]
mod tests_geometry;
#[cfg(test)]
mod tests_kernel_gates;
#[cfg(test)]
mod tests_oracle;
#[cfg(test)]
mod tests_prefill;

pub use band_codec::*;
pub use corpus_cache::*;
pub use decode::*;
pub use glue::*;
pub use launch::*;
pub use paged_geometry::*;
pub use prefill::*;
pub use synthetic::*;
