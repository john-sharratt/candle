//! A character's conversations (`docs/npc_conversations.md`): one-to-one,
//! parked exchanges with a person or another character, run beside the mind
//! loop as the same person.
//!
//! | Module | |
//! |---|---|
//! | [`authority`] | who may change the mission through a conversation |
//! | [`catchup`] | what the journal says happened since the last turn |
//! | [`clock`] | when a turn starts, and when an activation does |
//! | [`critical`] | a critical step: a request, then a yes |
//! | [`decide`] | where a decision's words go, and the decision restated |
//! | [`guard`] | the conversation's own guards |
//! | [`key`] | who a conversation is with, and its timeline |
//! | [`orders`] | its mission acts, checked and applied where they land |
//! | [`runner`] | every conversation held here, and whose turn is due |
//! | [`state`] | what a conversation keeps between turns |
//! | [`templates`] | the response templates a turn's words are shaped by |
//! | [`turn`] | one turn, end to end |
//! | [`voice`] | a channel post's words, decoded before the act is sent |
//! | [`within`] | what a conversation's turn can do |

pub mod authority;
pub mod catchup;
pub mod clock;
pub mod critical;
pub mod decide;
pub mod guard;
pub mod key;
pub mod orders;
pub mod runner;
pub mod state;
pub mod templates;
pub mod turn;
pub mod voice;
pub mod within;
