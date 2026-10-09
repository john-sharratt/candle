//! A character's journal: a short, dated, checked record of what it saw, heard,
//! did and left open, written off the main line and read back in the system
//! prompt. `docs/journal.md` is the design.
//!
//! | Module | Concern |
//! |---|---|
//! | [`entry`] | one entry, and how it reads back |
//! | [`state`] | what a character holds in memory, and when a draft is due |
//! | [`record`] | what each draft did, for the API |
//! | [`verify`] | the server's check of what the model wrote |
//! | [`section`] | the journal as system-prompt sections, and the record a restart rebuilds from |
//! | [`tools`] | the `journal_write` call, its grammar, and reading an answer |
//! | [`ask`] | what the character is asked to write about a stretch |
//! | [`workflow`] | one draft: the write, the check, the keep |
//! | [`desk`] | the character's live conversation a draft runs against |
//! | [`world`] | the world's facts a claim is checked against |
//!
//! Whether a stretch is worth an entry is asked by the guardian's journal module
//! (`engine::guardian::modules::journal`), which starts a draft on a yes.

pub mod ask;
pub mod closing;
pub mod desk;
pub mod entry;
pub mod record;
pub mod repeats;
pub mod section;
pub mod state;
pub mod tools;
pub mod verify;
pub mod workflow;
pub mod world;
