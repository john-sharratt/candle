//! A character's conversations (`docs/npc_conversations.md`): one-to-one,
//! parked exchanges with a person or another character, run beside the mind
//! loop as the same person.
//!
//! | Module | |
//! |---|---|
//! | [`clock`] | when a turn starts, and when an activation does |
//! | [`guard`] | the conversation's own guards |
//! | [`key`] | who a conversation is with, and its timeline |
//! | [`state`] | what a conversation keeps between turns |
//! | [`templates`] | the response templates a turn's words are shaped by |
//! | [`voice`] | a channel post's words, decoded before the act is sent |
//! | [`within`] | what a conversation's turn can do |

pub mod clock;
pub mod guard;
pub mod key;
pub mod state;
pub mod templates;
pub mod voice;
pub mod within;
