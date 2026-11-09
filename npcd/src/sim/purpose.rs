//! What a character means to tell somebody, on its way to the conversation
//! that will say it (`docs/npc_conversations.md` §10).
//!
//! A private `message` or a `reach_out` writes no line on a thread: the
//! character's conversation with that person says it, in its own words. The
//! act is performed on the world's authority, and the conversation runs
//! wherever the character is held, so the act leaves its purpose here and the
//! authority hands it on as the act lands.

use serde::{Deserialize, Serialize};

/// One purpose for one conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Purpose {
    /// The character that means it, by body id.
    pub from: String,
    /// The party it is for: a body id or an account handle.
    pub to: String,
    /// What it means to say.
    pub intent: String,
}
