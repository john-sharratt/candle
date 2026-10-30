//! The files of the repository, one module each (§3): `journal.yaml` and its
//! summary, `lines.yaml` and archives, `person.yaml` and `group.yaml`.

pub mod group;
pub mod lines;
pub mod meta;
pub mod person;
pub mod summary;

pub use group::{GroupFile, Kind, Member, Role, Who};
pub use lines::{read_lines, render_lines};
pub use meta::Meta;
pub use person::{Account, AccountKey, Email, PersonFile, MAX_FRAMED, MAX_GROUPS};
pub use summary::{Cite, Fingerprint, Sentence, Summary};
