//! A repository's git packs and refs as substrate records
//! (`docs/substrate_git.md`).
//!
//! A pack's bytes are a `Blob` record, named by their SHA-256. A small keyed
//! object, `git/pack/<repo>/<sha256>`, says the blob is a pack of that
//! repository; another, `git/ref/<repo>/<ref>`, says where a branch or tag
//! points and which packs it needs. Both replicate like every other record,
//! so every member holds every pack and agrees on every ref at each point in
//! the log.
//!
//! | Module | Concern |
//! |---|---|
//! | [`records`] | The records: what they hold and how they are named |
//! | [`store`] | The object store a member derives from them |
//! | [`publish`] | Writing a repository's new packs, then its refs |
//! | [`import`] | Bringing `origin`'s commits in, one member at a time |
//! | [`adopt`] | A member's first packs, from a checkout it already has |
//! | `substrate` | [`Records`] over `SharedSubstrate` (feature `substrate`) |

pub mod adopt;
pub mod import;
#[cfg(feature = "member")]
pub mod member;
pub mod publish;
pub mod records;
pub mod store;
#[cfg(feature = "substrate")]
pub mod substrate;

pub use adopt::adopt;
pub use import::{Imported, Importer};
pub use publish::{publish, Published};
pub use records::{Alone, Lease, Log, PackEntry, RefEntry, Records};
pub use store::{Store, Synced};
pub use records::import_lease;

/// The largest pack a member writes: one relocation stripe of the log
/// (`candle-conversation/src/persistence/stripes.rs`), and 64 replication
/// fragments.
pub const MAX_PACK: u64 = 64 * 1024 * 1024;
