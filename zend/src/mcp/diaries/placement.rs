//! Where a private conversation's diary calls are served
//! (`docs/private_mode.md` §5.4). What a conversation unlocked is held in one
//! member's memory alone, so every diary call of that conversation must reach
//! that member. The aggregator names the conversation's diary key in each
//! call's header, the diaries' route is sticky on it, and the member that
//! opened the diaries holds the key's claim until they are locked again — a
//! claim that, unlike a turn's, outlasts the request that took it.

use std::sync::{Arc, OnceLock};

use cluster::member::AppMember;
use mcp_server::refs::diary_key;

/// Keeps a conversation's diary calls on this member while its diaries are
/// open.
pub trait Placement: Send + Sync {
    /// Route `conversation`'s diary calls here, from now until
    /// [`Self::let_go`].
    fn hold(&self, conversation: &str);
    /// Stop routing them here: its diaries are locked.
    fn let_go(&self, conversation: &str);
}

static MEMBER: OnceLock<Arc<AppMember>> = OnceLock::new();

/// Place diary calls through `member`'s claims.
pub fn install(member: Arc<AppMember>) {
    let _ = MEMBER.set(member);
}

/// Placement through this member's cluster claims; nothing to place on a
/// member joined to no cluster, which serves every call itself.
pub struct Claims;

impl Placement for Claims {
    fn hold(&self, conversation: &str) {
        if let Some(member) = MEMBER.get() {
            member.hold(diary_key(conversation).as_bytes());
        }
    }

    fn let_go(&self, conversation: &str) {
        if let Some(member) = MEMBER.get() {
            // Nothing is replicated under a diary key, so whoever takes it
            // next waits for no record of this member's.
            member.let_go(diary_key(conversation).as_bytes(), 0);
        }
    }
}
