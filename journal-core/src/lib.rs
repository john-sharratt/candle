//! A working journal: what its writer saw, heard, did and worked out, written
//! down every so often by the writer itself and carried in its system prompt in
//! place of the history it covers.
//!
//! This crate is the journal's whole logic and none of its plumbing. A product
//! supplies where each turn came from ([`span::SpanTurn::evidence`]), what the
//! world reads when a claim is checked ([`verify::World`]), where the question is
//! put and the entry kept ([`workflow::Desk`]), and the wording it all reads in
//! ([`style::Style`]). npcd's characters and zend's coding sessions keep their
//! journals with it. The design is `docs/zend_journal.md`.

pub mod ask;
pub mod entry;
pub mod fold;
pub mod folding;
pub mod journal;
pub mod json_repair;
pub mod period;
pub mod record;
pub mod repeats;
pub mod section;
pub mod span;
pub mod style;
pub mod tools;
pub mod transcript;
pub mod verify;
pub mod workflow;
