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
//! submit until its last record is written: past its seal, until the
//! persistence writer has committed it ([`release`]). The claim therefore never
//! goes while a turn still writes the conversation, and the `through` it goes
//! with covers that turn.
//!
//! Once a claim goes, the member drops what it keeps of the conversation in
//! memory — the dialogue's sequence, the passthrough's live transcript. Its
//! next turn may be written elsewhere, and a sequence cached here would not
//! hold it: a conversation that moved away and back would be appended to as it
//! stood before it left. The next turn served here resumes from the substrate.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use cluster::member::{AppMember, ClaimUse};
use tokio::time::{sleep, Instant};

static MEMBER: Mutex<Option<Arc<AppMember>>> = Mutex::new(None);

/// Whether the conversation a claim key names has every turn this member
/// declared committed — installed with the handovers (`handover::install`).
type Committed = Box<dyn Fn(&str) -> bool + Send + Sync>;

static COMMITTED: OnceLock<Committed> = OnceLock::new();

/// How often a finished turn looks for its commit before letting its claim go.
const COMMIT_POLL: Duration = Duration::from_millis(20);

/// The longest a finished turn holds its claim waiting for its commit. The
/// persistence writer commits a sealed turn in milliseconds; past this the
/// writer is stalled, and the claim goes anyway — a conversation held for ever
/// is worse than one handed on a turn short.
const COMMIT_WAIT: Duration = Duration::from_secs(30);

/// Answer [`release`]'s question with `committed`: whether the conversation
/// a claim key names has every turn this member declared committed.
pub fn committed_by(committed: impl Fn(&str) -> bool + Send + Sync + 'static) {
    let _ = COMMITTED.set(Box::new(committed));
}

/// Let a finished turn's claim go once its conversation is committed.
///
/// **The seal is not the last record.** A turn seals in its own loop and the
/// persistence writer commits it after; a claim dropped between the two went
/// with the commit before — the previous turn's — as its `through`, and a
/// member holding everything through that took the conversation while the new
/// turn's snapshot had already reached it (`own_turns_committed`).
pub async fn release(claim: Option<ClaimUse>, key: &str) {
    let Some(claim) = claim else {
        return;
    };
    if let Some(committed) = COMMITTED.get() {
        let deadline = Instant::now() + COMMIT_WAIT;
        while !committed(key) {
            if Instant::now() >= deadline {
                tracing::warn!(
                    key,
                    waited_s = COMMIT_WAIT.as_secs(),
                    "a finished turn's commit never came; its claim goes without it"
                );
                break;
            }
            sleep(COMMIT_POLL).await;
        }
    }
    drop(claim);
}

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
