//! Where a residual lives between forwards — `docs/wave_step.md`.
//!
//! - [`row_map`]: caller order against a forward's internal order, and the
//!   copies that move rows between two layouts. Pure arithmetic.
//! - [`store`]: rows wherever they are held, and the copies between them.
//! - [`arena`]: rows held between steps in a slot tenant's arenas, compacted
//!   like every other tenant's.
//! - [`creep`]: a creep's rows, two halves flipped on commit, and their
//!   compaction.
//! - [`source`]: a forward's resumed residual, gathered into its own buffer.

pub mod arena;
pub mod creep;
pub mod row_map;
pub mod source;
pub mod store;

pub use arena::ResidualArena;
pub use creep::{compact_creeps, CreepCompaction, CreepResidual};
pub use row_map::{plan_copies, RowCopy, RowMap, Run};
pub use source::{move_rows, segment_parts, HandedOn, Part, ResidualIn};
pub use store::{copy_rows, RowStore, SpanRows};
