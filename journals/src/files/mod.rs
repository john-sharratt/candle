//! The files of the repository, one module each (§3): `journal.yaml` with
//! its references, focus and summary, each entry's own file, `person.yaml`,
//! `group.yaml`, and the README each group's folder opens with.

pub mod entry;
pub mod focus;
pub mod group;
pub mod meta;
pub mod person;
pub mod readme;
pub mod reference;
pub mod summary;

pub use entry::{entry_file, entry_id, in_order, read_entry};
pub use focus::Focus;
pub use group::{GroupFile, Member, Role, Who};
pub use meta::Meta;
pub use person::{Account, AccountKey, Email, PersonFile, MAX_GROUPS};
pub use readme::{group_readme, README};
pub use reference::{Ref, MAX_REFS};
pub use summary::{unmarked, Cite, Fingerprint, Sentence, Summary};
