//! Quantize in the gap between forwards (`docs/kv_quantize_in_the_gap.md`).
//!
//! The scheduler thread compresses the K/V a forward made in the gap before the
//! next one, on the compute stream: the host takes each sequence's completed
//! blocks (`table`), sizes the gap's demand per size class from the summaries
//! (`demand`), leases exactly that from the pools for the one gap (`lease`,
//! `lease_alloc`), and the device claims among the leased slots by select code
//! (`codes`) and converts in place (`launch`). At the next gap the catch-up
//! puts each converted block in place of its native chunk (`install`,
//! `assemble`).

// `assemble`, `codes`, `demand`, `install`, `launch`, `lease`, `lease_alloc` and
// `summary` are reached from their tests alone, so the dead-code lint is
// allowed over them.
#[allow(dead_code)]
mod assemble;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod codes;
#[allow(dead_code)]
mod demand;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod install;
#[cfg(all(test, feature = "cuda"))]
mod install_tests;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod launch;
#[cfg(all(test, feature = "cuda"))]
mod launch_tests;
#[allow(dead_code)]
mod lease;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod lease_alloc;
#[cfg(all(test, feature = "cuda"))]
mod lease_alloc_tests;
#[cfg(test)]
mod ownership_tests;
#[allow(dead_code)]
mod summary;
mod table;

pub use table::gap_take;
pub(crate) use table::{
    cursor_after_front_eviction, cursor_after_splice, cursor_after_truncate, LayerShape,
};
