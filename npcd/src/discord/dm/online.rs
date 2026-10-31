//! A bot as its DMs need it: online, and who it is there.

/// A character's bot whose shard Discord has answered `READY` on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Online {
    /// The character it is.
    pub npc: u64,
    /// The world the character has its body in, whose authority runs the bot.
    pub world: String,
    /// Its token, opened.
    pub token: String,
    /// Its own Discord user: what an echo of its own post is written by.
    pub user_id: u64,
    /// When Discord answered `READY`, in real milliseconds.
    pub since_ms: u64,
}
