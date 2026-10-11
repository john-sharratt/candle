//! Record identity: which substrate wrote a record, its place in that
//! substrate's sequence, and its version stamp (`docs/replicated_substrate.md`
//! §3.7, §6.2).

mod hlc;
mod machine_id;
mod node;
mod origination;
mod record_identity;
mod resident_ids;
#[cfg(test)]
mod substrate_tests;

pub use hlc::{wall_ms, Hlc, COUNTER_BITS};
pub use machine_id::machine_id;
pub use node::{fingerprint, load_or_mint, read as read_node, NodeSource, NODE_FILE};
pub use origination::{now_ms, Origination};
pub use record_identity::{RecordIdentity, VersionStamp, GEN_DETACHED};
pub use resident_ids::ResidentIds;
