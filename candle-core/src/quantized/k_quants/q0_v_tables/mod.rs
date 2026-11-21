//! The Q0_V codebook tables: the curve, scale and centroid tables the codec indexes,
//! one set for K and one for V.
//!
//! The CUDA side in `q0_v_tables.cuh` holds the same tables, so the Rust reference
//! codec and the kernels decode a block alike. K and V are calibrated separately,
//! and the call site picks a side by `IS_K`. The 16-entry centroid tables are built
//! from the 8-entry calibration, and a full recalibration has not been measured.
//!
//! - `curve_picks`: the 128-entry K and V curve tables.
//! - `scales_centroids`: the scale and centroid tables and peak indices.
//! - `curve_full_k`: the full K-side curve calibration.
//! - `curve_full_v`: the full V-side curve calibration.

mod curve_full_k;
mod curve_full_v;
mod curve_picks;
mod scales_centroids;

pub use curve_picks::*;
pub use scales_centroids::*;
