//! The command table's mission generator: work on the record, found in the
//! corpus and written up as missions a Maker can carry out.
//!
//! The Makers exist to finish a world's storyline — to correct where the record
//! contradicts itself, to tell what it passes over, and to give every character
//! a life, one significant event at a time. The routine bank
//! ([`crate::engine::mission::bank`]) cannot set that work: it knows the rooms
//! and the people, not what the record lacks. This does.
//!
//! - [`config`] — `<mind>/missions.yaml`: the generators, their prompts and
//!   weights, and how many missions wait at the table.
//! - [`corpus`] — the mind as the generator reads it: eras, lives, stories.
//! - [`target`] — the kinds of work, and the next piece of the corpus each is
//!   about, chosen by the engine against the ledger.
//! - [`material`] — what the model is shown about a target.
//! - [`answer`] — the `mission` / `no_mission` call, checked against the corpus
//!   and made into a [`crate::engine::mission::Mission`].
//! - [`run`] — one generation, and the loop that keeps the table stocked.
//!
//! A generated mission carries its work ([`crate::engine::mission::Work`]): the
//! document to write and those to read. The engine signs off the reads and the
//! write as they happen at a bench, and the report waits until the write is
//! committed — so a mission done is a document on the record, not a claim.
//!
//! The design is `docs/mission_generator.md`.

pub mod answer;
pub mod canon;
pub mod config;
pub mod copied;
pub mod corpus;
pub mod fingerprint;
pub mod gates;
pub mod glossary;
pub mod leakage;
pub mod material;
pub mod reading;
pub mod rejection;
#[cfg(test)]
mod replay;
pub mod research;
pub mod run;
pub mod step;
pub mod target;
