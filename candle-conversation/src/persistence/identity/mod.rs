//! Record identity: which substrate wrote a record, its place in that
//! substrate's sequence, and its version stamp (`docs/replicated_substrate.md`
//! §3.7, §6.2).

mod hlc;
mod machine_id;
mod marker_ids;
mod minter;
mod node;
mod origination;
mod record_identity;
mod section_generations;
#[cfg(test)]
mod substrate_tests;

pub use hlc::{wall_ms, Hlc, COUNTER_BITS};
pub use machine_id::machine_id;
pub use marker_ids::{is_marker, MarkerIds};
pub use minter::Minter;
pub use node::{fingerprint, load_or_mint, read as read_node, NodeSource, NODE_FILE};
pub use origination::{now_ms, Origination};
pub use record_identity::{RecordIdentity, Stamped, VersionStamp, GEN_DETACHED};
pub use section_generations::{OwnSections, SectionGenerations};
