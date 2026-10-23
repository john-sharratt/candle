//! A character's journal: a short, dated, checked record of what it saw, heard,
//! did and left open, written off the main line and read back in the system
//! prompt. `docs/journal.md` is the design.
//!
//! The journal itself — entries, the check, the write grammar, the draft — is
//! `journal_core`'s. What is the character's is here:
//!
//! | Module | Concern |
//! |---|---|
//! | [`state`] | the character's journal, and when a stretch of its window is due |
//! | [`origin`] | a window turn as a stretch holds it, and what a claim citing it rests on |
//! | [`style`] | the world calendar and the character's voice an entry reads in |
//! | [`section`] | which conversation holds a character's journal |
//! | [`closing`] | the entry that closes a mission's chapter |
//! | [`desk`] | the character's live conversation a draft runs against |
//! | [`world`] | the world's facts a claim is checked against |
//!
//! Whether a stretch is worth an entry is asked by the guardian's journal module
//! (`engine::guardian::modules::journal`), which starts a draft on a yes.

pub mod closing;
pub mod desk;
pub mod origin;
pub mod section;
pub mod state;
pub mod style;
pub mod world;
