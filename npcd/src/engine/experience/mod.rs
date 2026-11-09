//! A character's experience as journal lines (`docs/npc_conversations.md`
//! §5): what it heard, said, wrote, promised, did, and what changed for it,
//! written into `groups/<group>/bots/<bot>/` of its world's group.
//!
//! A conversation knows what happened since it last spoke with its party
//! only from these lines (§6.1), and the main loop knows what its
//! conversations did from them (§5.4).
//!
//! | Module | |
//! |---|---|
//! | [`buffer`] | the unwritten lines, per bot, and their caps |
//! | [`concern`] | which journals hear of a change in the world |
//! | [`line`] | how each kind of line reads |
//! | [`store`] | the checkout, the per-world groups, the flush |

pub mod buffer;
pub mod concern;
pub mod line;
pub mod store;
