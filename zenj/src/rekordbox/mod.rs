//! rekordbox's own records, read-only (§4.1, step 5): the library
//! ([`masterdb`], decrypted by [`sqlcipher`]) and each track's analysis
//! ([`anlz`]) — its grid, phrases and cues — read into what a profile needs
//! ([`analysis`]).

pub mod analysis;
pub mod anlz;
pub mod masterdb;
pub mod sqlcipher;

pub use analysis::{analyse, Analysis};
pub use masterdb::Content;
