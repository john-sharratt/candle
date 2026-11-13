//! Quantize in the gap between forwards (`docs/kv_quantize_in_the_gap.md`).
//!
//! The scheduler thread compresses the K/V a forward made in the gap before the
//! next one, on the compute stream: the host takes each sequence's completed
//! blocks (`table`), leases destination slots per size class (`lease`), and the
//! device claims among them by select code (`codes`) and converts in place.

// `launch`, `lease` and `summary` are reached from their tests alone, so the
// dead-code lint is allowed over them.
#[cfg(feature = "cuda")]
mod codes;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod launch;
#[cfg(all(test, feature = "cuda"))]
mod launch_tests;
#[allow(dead_code)]
mod lease;
#[allow(dead_code)]
mod summary;
mod table;

pub use table::gap_take;
pub(crate) use table::{
    cursor_after_front_eviction, cursor_after_splice, cursor_after_truncate, LayerShape,
};
