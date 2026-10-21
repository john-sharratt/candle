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

/// How often a finished turn looks for its commit at first, while the writer
/// is likely to land it any moment.
const COMMIT_POLL: Duration = Duration::from_millis(20);

/// How long it looks that often before it settles to [`COMMIT_POLL_SLOW`].
const COMMIT_POLL_FAST_FOR: Duration = Duration::from_secs(2);

/// How often it looks once the writer is plainly behind.
const COMMIT_POLL_SLOW: Duration = Duration::from_millis(250);

/// How often a turn still waiting says so.
const COMMIT_WARN_EVERY: Duration = Duration::from_secs(30);

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
///
/// **It waits as long as the writer takes, and never gives up.** Under load the
/// writer's queue runs past half a minute (run 41: eight turns, on both
/// members); a release on a deadline named the previous commit again, which is
/// the race this exists to close. Holding the claim longer costs only where the
/// next turn may be placed — this member keeps serving the conversation — so
/// the wait is unbounded and says so every [`COMMIT_WARN_EVERY`]. The caller
/// ends the reply's stream first, so no client waits on it.
pub async fn release(claim: Option<ClaimUse>, key: &str) {
    let Some(claim) = claim else {
        return;
    };
    if let Some(committed) = COMMITTED.get() {
        let started = Instant::now();
        let mut warned = started;
        while !committed(key) {
            let waited = started.elapsed();
            if warned.elapsed() >= COMMIT_WARN_EVERY {
                warned = Instant::now();
                tracing::warn!(
                    key,
                    waited_s = waited.as_secs(),
                    "a finished turn still waits for its commit; its claim is held until it lands"
                );
            }
            sleep(if waited < COMMIT_POLL_FAST_FOR {
                COMMIT_POLL
            } else {
                COMMIT_POLL_SLOW
            })
            .await;
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
