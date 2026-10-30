//! The files of the repository, one module each (§3): `journal.yaml` and its
//! summary, each entry's own file, `person.yaml`, `group.yaml`, and the README
//! each owner's folder opens with.

pub mod entry;
pub mod group;
pub mod meta;
pub mod person;
pub mod readme;
pub mod summary;

pub use entry::{entry_file, entry_id, in_order, read_entry};
pub use group::{GroupFile, Kind, Member, Role, Who};
pub use meta::Meta;
pub use person::{Account, AccountKey, Email, PersonFile, MAX_FRAMED, MAX_GROUPS};
pub use readme::{group_readme, person_readme, README};
pub use summary::{Cite, Fingerprint, Sentence, Summary};
