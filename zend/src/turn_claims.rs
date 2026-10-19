//! A conversation is this member's for one turn
//! (`docs/cooperative_admission.md` §2.2).
//!
//! The member's claims follow use: a conversation's claim lasts while something
//! uses it and goes with the last use, announced with this member's own `oseq`
//! as it stands then. Its next turn is placed afresh, among the members that
//! have applied this one.
//!
//! A request uses the claim until its response ends (the host's sticky
//! check). A turn outlives its request when the client goes away — it runs its
//! tools and seals for a reload to find — so a turn uses the claim too, from
//! submit until its last record is written. The claim therefore never goes
//! while a turn still writes the conversation, and the `through` it goes with
//! covers that turn.
//!
//! Once a claim goes, the member drops what it keeps of the conversation in
//! memory — the dialogue's sequence, the passthrough's live transcript. Its
//! next turn may be written elsewhere, and a sequence cached here would not
//! hold it: a conversation that moved away and back would be appended to as it
//! stood before it left. The next turn served here resumes from the substrate.

use std::sync::{Arc, Mutex};

use cluster::member::{AppMember, ClaimUse};

static MEMBER: Mutex<Option<Arc<AppMember>>> = Mutex::new(None);

/// Place every conversation turn by turn on `member`. `forget` is told each
/// conversation this member stops holding: from then on another member may
/// write its next turn, so what this one keeps of it in memory must go.
pub fn install(member: Arc<AppMember>, forget: impl Fn(&str) + Send + Sync + 'static) {
    member.follow_use();
    member.on_release(Box::new(move |key| forget(&String::from_utf8_lossy(key))));
    *MEMBER.lock().unwrap_or_else(|e| e.into_inner()) = Some(member);
}

/// The turn on the conversation `key` names — its `conv_id`, or a passthrough
/// transcript's key — uses its claim until the returned guard drops. `None`
/// on a member not joined to a cluster, where there is no claim.
pub fn hold(key: &str) -> Option<ClaimUse> {
    let member = MEMBER.lock().unwrap_or_else(|e| e.into_inner()).clone()?;
    Some(member.use_claim(key.as_bytes()))
}
