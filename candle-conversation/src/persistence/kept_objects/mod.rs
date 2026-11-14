//! A substrate's kept custom objects as text, and back
//! (`docs/hit_driven_projection.md` §12 item 17, §13.4).
//!
//! A substrate moved aside for a new model key keeps what is not
//! model-derived: accounts, journals' records, MCP state, a character's world.
//! Its daemon exports those objects — and the blobs they name, and the
//! tombstones it holds — as JSON Lines, and the new substrate imports the
//! union of every member's export, the latest original stamp winning per key.
//!
//! | Module | Concern |
//! |---|---|
//! | [`classify`] | Which keys are kept and which are left out |
//! | [`format`] | The header line, the object and blob lines, and their stamp |
//! | [`names`] | The blobs a kept object names |
//! | [`export`] | A substrate's objects as an export |
//! | [`union`] | Several exports as one set of winners |
//! | [`import`] | A union written into a substrate |

pub mod classify;
pub mod export;
pub mod format;
pub mod import;
pub mod names;
pub mod union;

pub use classify::{is_kept, LEFT_OUT};
pub use export::export;
pub use format::{Header, Line, Stamp, FORMAT, VERSION};
pub use import::{import, ImportReport};
pub use union::{union, Union};
