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

use cluster::member::{AppMember, Check, ClaimUse};
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

/// How long a turn waits before it says it is waiting — once, and again when its
/// commit lands.
const COMMIT_WAIT_NOTED: Duration = Duration::from_secs(30);

/// Answer [`release`]'s question with `committed`: whether the conversation
/// a claim key names has every turn this member declared committed.
pub fn committed_by(committed: impl Fn(&str) -> bool + Send + Sync + 'static) {
    let _ = COMMITTED.set(Box::new(committed));
}

/// Whether the conversation a claim key names was let go by another member
/// through records this member has not yet applied — installed with the
/// handovers (`handover::install`).
type Awaited = Box<dyn Fn(&str) -> bool + Send + Sync>;

static AWAITED: OnceLock<Awaited> = OnceLock::new();

/// Answer [`take`]'s question with `awaited`.
pub fn awaited_by(awaited: impl Fn(&str) -> bool + Send + Sync + 'static) {
    let _ = AWAITED.set(Box::new(awaited));
}

/// Whether this member still awaits the records the conversation `key` was
/// last let go through.
fn awaiting(key: &str) -> bool {
    AWAITED.get().is_some_and(|awaited| awaited(key))
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
/// the wait is unbounded. A wait past [`COMMIT_WAIT_NOTED`] says so once, and
/// once more with its length when the commit lands — a line every 30 s per turn
/// was 957 lines in one run. The caller ends the reply's stream first, so no
/// client waits on it.
pub async fn release(claim: Option<ClaimUse>, key: &str) {
    let Some(claim) = claim else {
        return;
    };
    if let Some(committed) = COMMITTED.get() {
        let started = Instant::now();
        let mut noted = false;
        while !committed(key) {
            let waited = started.elapsed();
            if !noted && waited >= COMMIT_WAIT_NOTED {
                noted = true;
                tracing::warn!(
                    key,
                    waited_s = waited.as_secs(),
                    "a finished turn waits for its commit; its claim is held until it lands"
                );
            }
            sleep(if waited < COMMIT_POLL_FAST_FOR {
                COMMIT_POLL
            } else {
                COMMIT_POLL_SLOW
            })
            .await;
        }
        if noted {
            tracing::info!(
                key,
                waited_s = started.elapsed().as_secs(),
                "a finished turn's commit landed; its claim goes"
            );
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

/// Whether another live member holds the conversation `key` names: something
/// there — a turn, a request answering — is using it now. `false` on a member
/// not joined to a cluster.
pub fn held_elsewhere(key: &str) -> bool {
    let Some(member) = MEMBER.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
        return false;
    };
    member
        .holder_of(key.as_bytes())
        .is_some_and(|holder| holder != member.node())
}

/// How often [`take`] looks for the member's admission.
const ADMISSION_POLL: Duration = Duration::from_millis(250);

/// Take the conversation `key` names for a turn no request brought here — one
/// resumed after a restart — and use the claim until the guard drops.
///
/// **A request's claim is taken by the host's sticky check before the turn
/// runs; a resumed turn has no request, so it takes its own.** Without it a
/// resumed round ran with no claim at all: the gateway sent the conversation's
/// page and next turn to whichever member it liked, and a member restarting
/// beside it read the replicated resume entry of the round still running and
/// ran it a second time. On 2026-11-09 both members ran conversation 425bad8d
/// at once, one sealing turns 85–87 over the other's recurrent state parked at
/// 84, whose next turn then failed reading it back.
///
/// **And it takes only a conversation it holds complete.** A request reaches a
/// member the gateway placed it on, and after a release the gateway places it
/// only on a member that has reported holding the conversation complete
/// through the release (`handover`). A resumed turn is placed by nobody, so it
/// asks the same thing itself: a member that took a conversation let go a
/// moment ago, its last turns not yet applied here, would resume a round its
/// releaser had already finished, from history that stops short of it — the
/// duplicate run of 2026-11-09 was working from exactly such a copy, and
/// believed the last question asked was a different one.
///
/// `Err` when another live member holds the claim — that member is writing
/// the conversation, and this one must leave it alone — or when this member
/// still lacks the records it was let go through after [`COMPLETE_WAIT`].
/// Waits for the member's admission first, because the claims are learned
/// there — before it every key reads as free. `Ok(None)` on a member not
/// joined to a cluster.
pub async fn take(key: &str) -> Result<Option<ClaimUse>, NotTaken> {
    let Some(member) = MEMBER.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
        return Ok(None);
    };
    while !member.admitted() {
        sleep(ADMISSION_POLL).await;
    }
    let started = Instant::now();
    while awaiting(key) {
        if started.elapsed() >= COMPLETE_WAIT {
            return Err(NotTaken::Incomplete);
        }
        sleep(ADMISSION_POLL).await;
    }
    match member.claim(key.as_bytes()) {
        Check::Held(holder) => Err(NotTaken::Held(holder)),
        Check::Mine | Check::Took(_) => Ok(Some(member.use_claim(key.as_bytes()))),
    }
}

/// How long [`take`] waits for the records a conversation was let go through.
const COMPLETE_WAIT: Duration = Duration::from_secs(60);

/// Why [`take`] did not take a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotTaken {
    /// Another live member, this node, holds it.
    Held(u64),
    /// It was let go through records this member has not applied.
    Incomplete,
}
