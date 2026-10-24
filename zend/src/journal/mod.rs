//! A dialogue's working journal: short, dated, cited notes of what the session
//! saw, was told, did and left open, written every so often by the model itself
//! and carried in its system prompt beside the turns the projection still holds.
//! `docs/zend_journal.md` is the design.
//!
//! The journal itself — entries, the check, the write grammar, the draft — is
//! `journal_core`'s, shared with npcd. What is a coding session's is here:
//!
//! | Module | Concern |
//! |---|---|
//! | [`origin`] | the dialogue's turns as citable parts, and what each supports |
//! | [`state`] | a session's journal and its cadence |
//! | [`pressure`] | whether the projection has started leaving out unread turns |
//! | [`style`] | the UTC dates and coding words an entry reads in |
//! | [`world`] | the workspace a typed claim is checked against |
//! | [`desk`] | the draft's question put to a focused fork of the dialogue, at `Normal` |
//! | [`draft`] | one draft — write or skip in one answer — and its record |
//! | [`carry`] | the journal as the system prompt carries it |
//! | [`record`] | the record a restart rebuilds the journal from |
//! | [`text`] | the turns behind each entry, whole, for `journal_read` and `journal_grep` |
//! | [`registry`] | every dialogue's journal, by conversation |
//!
//! Only a dialogue journals. A journal is private to its conversation: no other
//! conversation retrieves its sections and no layer reads them, and it is
//! tombstoned with the conversation.

pub mod carry;
pub mod desk;
pub mod draft;
pub mod origin;
pub mod pressure;
pub mod record;
pub mod registry;
pub mod state;
pub mod style;
pub mod text;
pub mod world;
