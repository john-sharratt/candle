//! The QSA index page cache (`docs/hit_driven_projection.md` §7.6): the index
//! pages the pools read, held in one host-memory slab taken at startup, pinned
//! where the driver grants it, shared by every substrate in the process.
//!
//! A page is read from the log straight into its block. It is found again by
//! its stream and record identity until it is evicted, least recently used
//! first, and never while anything holds it.

mod cache;
mod cached;
mod classes;
mod install;
mod slab;

pub use cache::{CacheStats, IndexPageCache};
pub use cached::CachedBytes;
#[cfg(test)]
pub(crate) use install::install_for_tests;
pub use install::{global, install};
