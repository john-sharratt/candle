//! Decode-time health checks that detect degenerate generation and log why they
//! fired.
//!
//! Repetition and phrase-loop checks run every step on the CPU. The NaN, infinity,
//! magnitude and entropy checks synchronise the GPU at the logit check interval,
//! so their cost is paid only where it buys a signal. Every warning is emitted
//! under a fixed tracing target, so a layer can capture health events separately
//! from the general log. `DecodeHealthConfig` is always compiled so callers can
//! set it, while the checks themselves sit behind the `decode-health` feature.
//!
//! - `dump`: the post-abort trajectory report.
//! - `entropy`: the entropy collapse checks.
//! - `events`: the health event types.
//! - `state`: the per-decode state and its log.

use std::collections::VecDeque;

use candle::Tensor;
use candle_nn::ops::softmax;

mod dump;
mod entropy;
mod events;
mod state;

pub use dump::*;
pub use entropy::*;
pub use events::*;
pub use state::*;
