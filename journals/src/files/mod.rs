//! The files of the repository, one module each (§3): `journal.yaml` and its
//! summary, `lines.yaml` and archives, `person.yaml` and `family.yaml`.

pub mod family;
pub mod lines;
pub mod meta;
pub mod person;
pub mod summary;

pub use family::{FamilyFile, Member, Role};
pub use lines::{read_lines, render_lines};
pub use meta::Meta;
pub use person::{Account, AccountKey, Email, PersonFile, MAX_FAMILIES};
pub use summary::{Cite, Fingerprint, Sentence, Summary};
