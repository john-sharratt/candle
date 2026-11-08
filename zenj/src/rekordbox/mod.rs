//! rekordbox: its own records, read-only (§4.1, step 5) — the library
//! ([`masterdb`], decrypted by [`sqlcipher`]) and each track's analysis
//! ([`anlz`]), its grid, phrases and cues, read into what a profile needs
//! ([`analysis`]) — and the application itself, opened and closed for the
//! house ([`process`], [`control`]).

pub mod analysis;
pub mod anlz;
pub mod control;
pub mod masterdb;
pub mod process;
pub mod sqlcipher;

pub use analysis::{analyse, Analysis};
pub use masterdb::Content;
