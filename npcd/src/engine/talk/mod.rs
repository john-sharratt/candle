//! A character's conversations (`docs/npc_conversations.md`): one-to-one,
//! parked exchanges with a person or another character, run beside the mind
//! loop as the same person.
//!
//! | Module | |
//! |---|---|
//! | [`templates`] | the response templates a turn's words are shaped by |
//! | [`voice`] | a channel post's words, decoded before the act is sent |

pub mod templates;
pub mod voice;
