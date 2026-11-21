//! Tensor operations for the layers of this crate: activations, attention, dropout,
//! normalisation and softmax.
//!
//! Each operation is either a composition of tensor operations or a fused custom op
//! with a reference composition. Operations that produce a new tensor place it in the
//! arena of their input, so a chain of operations stays in one memory region unless a
//! caller names another.
//!
//! - `activations`: elementwise activations and the fused SiLU-multiply.
//! - `attention`: fused scaled dot-product attention.
//! - `layers`: dropout, pixel shuffle and padding helpers, and the identity layer.
//! - `norms`: RMS and layer normalisation.
//! - `softmax_ops`: softmax and log-softmax.

use candle::wave_provenance::WaveTicket;
use candle::{CpuStorage, DType, Layout, LiveTensor, Module, Result, Shape, Tensor, D};
use rayon::prelude::*;

mod activations;
mod attention;
mod layers;
mod norms;
mod softmax_ops;

pub use activations::*;
pub use attention::*;
pub use layers::*;
pub use norms::*;
pub use softmax_ops::*;
