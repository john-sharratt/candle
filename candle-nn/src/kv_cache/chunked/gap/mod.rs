//! Quantize in the gap between forwards (`docs/kv_quantize_in_the_gap.md`).
//!
//! The scheduler thread compresses the K/V a forward made in the gap before the
//! next one, on the compute stream: the host takes each sequence's completed
//! blocks (`table`), sizes the gap's demand per size class from the summaries
//! (`demand`), leases exactly that from the pools for the one gap (`lease`,
//! `lease_alloc`), and the device claims among the leased slots by select code
//! (`codes`) and converts in place (`launch`). At the next gap the catch-up
//! puts each converted block in place of its native chunk (`install`,
//! `assemble`). [`GapCycle`] keeps all of it from one gap to the next.

// The host arithmetic (`assemble`, `demand`, `lease`, `summary`) is read only
// by the device path; built without a device for its own tests alone.
#[cfg(any(feature = "cuda", test))]
mod assemble;
#[cfg(feature = "cuda")]
mod codes;
#[cfg(feature = "cuda")]
mod cycle;
#[cfg(all(test, feature = "cuda"))]
mod cycle_tests;
#[cfg(any(feature = "cuda", test))]
mod demand;
#[cfg(feature = "cuda")]
mod install;
#[cfg(all(test, feature = "cuda"))]
mod install_tests;
#[cfg(feature = "cuda")]
mod launch;
#[cfg(all(test, feature = "cuda"))]
mod launch_tests;
#[cfg(any(feature = "cuda", test))]
mod lease;
#[cfg(feature = "cuda")]
mod lease_alloc;
#[cfg(all(test, feature = "cuda"))]
mod lease_alloc_tests;
#[cfg(test)]
mod ownership_tests;
#[cfg(all(test, feature = "cuda"))]
mod sign_tests;
#[cfg(any(feature = "cuda", test))]
mod summary;
mod table;

#[cfg(feature = "cuda")]
pub use cycle::{CatchUpHost, GapCatchUp, GapCycle, GapQuantize, GapSeq, QuantizeHost};
#[cfg(feature = "cuda")]
pub use demand::GapKind;
#[cfg(feature = "cuda")]
pub use launch::gap_jobs_within;
#[cfg(feature = "cuda")]
pub use lease_alloc::LeaseRefusal;
pub use table::gap_take;
pub(crate) use table::{
    cursor_after_front_eviction, cursor_after_splice, cursor_after_truncate, LayerShape,
};
