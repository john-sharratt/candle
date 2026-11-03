//! The memory census (`docs/memory_census.md`): what a process holds, counted
//! from inside it, with no debugger and no trace.
//!
//! | Module | What |
//! |---|---|
//! | [`classes`] | Size classes: the power of two a request rounds up to |
//! | [`shards`] | The census's counters, one shard per thread |
//! | [`scope`] | Which part of the daemon a block was made in, named in the code |
//! | [`counting`] | [`Counting`], the global allocator that counts |
//! | [`os`] | The OS's figures: commit charge, the heap's summary, the machine's commit |
//! | [`reading`] | One reading of all of it |
//! | [`history`] | An hour of readings |
//! | [`verdict`] | Steady, retention, an owner, or outside the heap |
//! | [`report`] | The census as `/v1/memory` carries it |
//!
//! A binary installs the allocator with
//!
//! ```ignore
//! #[global_allocator]
//! static HEAP: heap_census::Counting = heap_census::Counting;
//! ```
//!
//! Without it the OS's figures are still read, and a verdict still drawn from
//! the heap's own counts; only the size classes are missing.

pub mod classes;
pub mod counting;
pub mod history;
pub mod os;
pub mod reading;
pub mod report;
pub mod scope;
pub mod shards;
pub mod verdict;

pub use counting::Counting;
pub use scope::{enter, set_thread, Scope};
