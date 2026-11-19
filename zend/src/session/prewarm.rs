//! The journal pre-warmer (`docs/shared_journal_sections.md` §7).
//!
//! A new conversation's journal opens with a shared run — its ancestors'
//! journals, the entries' heading, the opening, the close — that every new
//! conversation of the same scope carries word for word within an hour. This
//! seals that run ahead of the conversation, through the same reconcile a
//! conversation runs, so the conversation's first round restores it instead of
//! prefilling it. A run is sealed again when what it is made of moves: an
//! ancestor's journal, what the priming chain read, the hour.
//!
//! The run is held by nothing once it is durable: the warmer's conversation
//! lets go of it and is tombstoned, and the substrate alone keeps it, hot
//! nowhere until a conversation reads it.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use candle_conversation::persistence::streams::StreamId;
use candle_conversation::projection::{DecodePriority, TimelineId};
use candle_conversation::{Carried, Linking, Sequence, Sharing, Want};
use journal_core::journal::Holds;
use journal_core::section::{Section, JOURNAL};

use super::{journal_sections, InferenceState};
use crate::ingest_lease::{share_prewarm, Share};
use crate::journal::state::SessionJournal;
use crate::journal::style::style_for;
use crate::layers::expiry::now_ms;
use crate::mcp::ownership::owner_of;

/// How often the warmer looks for a run to seal.
const PREWARM_EVERY: Duration = Duration::from_secs(60);

/// How long a scope stays warm after a conversation of it last carried its
/// journal.
const SCOPE_TTL: Duration = Duration::from_secs(24 * 3600);

/// How far ahead of the hour the next hour's run is sealed, so the first
/// conversation after it hits.
const HOUR_LEAD_MS: u64 = 5 * 60_000;

const HOUR_MS: u64 = 3_600_000;

/// How long a sealed run is given to become durable before it is let go.
const DURABLE_WAIT: Duration = Duration::from_secs(60);

/// How long a run is given to seal.
const SEAL_WAIT: Duration = Duration::from_secs(120);

/// The thread a scope's run is sealed for: one no conversation is in, so it
/// carries no thread journal and no memory, as a new thread does not.
const FRESH_THREAD: &str = "~prewarm";

/// Whose new conversations carry one shared run.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Scope {
    /// A new conversation of an owner's: the owner's project journals ahead of
    /// the opening, and the reader's own line after it.
    Owner(String),
    /// A conversation owned through no thread: the opening alone.
    Threadless,
}

impl Scope {
    /// The scope a conversation named `conv_id` is in.
    pub(super) fn of(conv_id: &str) -> Scope {
        match owner_of(conv_id) {
            Some(owner) => Scope::Owner(owner.to_string()),
            None => Scope::Threadless,
        }
    }

    /// The name its lease is taken under.
    fn key(&self) -> String {
        match self {
            Scope::Owner(owner) => format!("owner/{owner}"),
            Scope::Threadless => "threadless".to_string(),
        }
    }

    /// A conversation id a new conversation of the scope could have: what its
    /// journal sections are built for.
    fn conv_id(&self) -> String {
        match self {
            Scope::Owner(owner) => format!("mcp:{owner}:{FRESH_THREAD}/zen"),
            Scope::Threadless => "prewarm".to_string(),
        }
    }
}

/// The times a run is dated at now: this hour's, and the next hour's once it
/// is within [`HOUR_LEAD_MS`].
pub(super) fn warm_times(now_ms: u64) -> Vec<u64> {
    let next = (now_ms / HOUR_MS + 1) * HOUR_MS;
    let mut at = vec![now_ms];
    if next - now_ms <= HOUR_LEAD_MS {
        at.push(next);
    }
    at
}

/// The hour `at_ms` is in.
fn hour_of(at_ms: u64) -> u64 {
    at_ms / HOUR_MS
}

/// What the warmer knows: the scopes conversations carried a journal in, and
/// for each scope and hour the run it sealed and the streams it left.
#[derive(Default)]
pub(super) struct Prewarm {
    seen: Mutex<HashMap<Scope, Instant>>,
    sealed: Mutex<HashMap<(Scope, u64), Sealed>>,
}

/// One run the warmer sealed: what it was made of, and its streams.
#[derive(Clone)]
struct Sealed {
    digest: u64,
    streams: Vec<StreamId>,
}

impl Prewarm {
    /// A conversation named `conv_id` carried its journal at `at`.
    pub(super) fn note(&self, conv_id: &str, at: Instant) {
        self.seen.lock().unwrap().insert(Scope::of(conv_id), at);
    }

    /// The scopes of the conversations the substrate holds, seen now: what a
    /// start warms before any conversation carries its journal again.
    pub(super) fn seed(&self, known: &[(TimelineId, String, String, bool, u64)]) {
        let now = Instant::now();
        let mut seen = self.seen.lock().unwrap();
        for (_, conv_id, _, archived, _) in known {
            if !archived {
                seen.entry(Scope::of(conv_id)).or_insert(now);
            }
        }
    }

    /// The scopes to keep warm at `now`: every one seen within [`SCOPE_TTL`],
    /// and the threadless one always.
    fn due(&self, now: Instant) -> Vec<Scope> {
        let mut seen = self.seen.lock().unwrap();
        seen.retain(|_, at| now.saturating_duration_since(*at) < SCOPE_TTL);
        let mut due: Vec<Scope> = seen.keys().cloned().collect();
        if !due.contains(&Scope::Threadless) {
            due.push(Scope::Threadless);
        }
        due.sort_by_key(Scope::key);
        due
    }

    /// Whether the run `digest` names is the one sealed for `scope` at `hour`.
    fn is_sealed(&self, scope: &Scope, hour: u64, digest: u64) -> bool {
        self.sealed
            .lock()
            .unwrap()
            .get(&(scope.clone(), hour))
            .is_some_and(|s| s.digest == digest)
    }

    fn record(&self, scope: Scope, hour: u64, sealed: Sealed) {
        self.sealed.lock().unwrap().insert((scope, hour), sealed);
    }

    /// Forget the runs of hours before `hour`: their streams are the sweep's.
    fn forget_before(&self, hour: u64) {
        self.sealed.lock().unwrap().retain(|(_, h), _| *h >= hour);
    }

    /// Every stream a current run holds.
    fn current(&self) -> HashSet<StreamId> {
        self.sealed
            .lock()
            .unwrap()
            .values()
            .flat_map(|s| s.streams.iter().copied())
            .collect()
    }
}

/// What a run is made of, as one number: its sections' names, texts and
/// sharing, in order.
fn digest(sections: &[Section], sharing: &[Sharing]) -> u64 {
    let mut h = DefaultHasher::new();
    for (s, sharing) in sections.iter().zip(sharing) {
        s.name.hash(&mut h);
        s.text.hash(&mut h);
        format!("{sharing:?}").hash(&mut h);
    }
    h.finish()
}

/// The shared run a new conversation of `scope` carries at `at_ms`, and each
/// section's sharing: the journal sections a conversation builds, with the
/// opening dated `at_ms`, up to the first that is the conversation's own.
fn run_of(state: &InferenceState, scope: &Scope, at_ms: u64) -> (Vec<Section>, Vec<Sharing>) {
    let conv_id = scope.conv_id();
    let mut journal = SessionJournal::new().in_style(style_for(&conv_id));
    for entry in state.branch_ingest.seed_read().entries(at_ms) {
        journal.journal_mut().keep_closing(entry);
    }
    let (_, sections, sharing) = journal_sections(state, &conv_id, &Mutex::new(journal));
    let run = sharing.iter().take_while(|s| s.is_shared()).count();
    (
        sections.into_iter().take(run).collect(),
        sharing.into_iter().take(run).collect(),
    )
}

/// The streams of the shared sections no current run holds and no live section
/// entry reads: what the sweep retires.
pub(super) fn retirable(
    shared: &[StreamId],
    current: &HashSet<StreamId>,
    held: &HashSet<StreamId>,
) -> Vec<StreamId> {
    shared
        .iter()
        .copied()
        .filter(|s| !current.contains(s) && !held.contains(s))
        .collect()
}

/// The warmer's loop. Holds the state weakly and ends with it.
pub(super) async fn run(state: Weak<InferenceState>) {
    let mut every = tokio::time::interval(PREWARM_EVERY);
    loop {
        every.tick().await;
        let Some(state) = state.upgrade() else {
            return;
        };
        pass(&state).await;
    }
}

/// One pass: seal every due scope's run that moved, then retire what no run
/// holds.
async fn pass(state: &Arc<InferenceState>) {
    let now = now_ms();
    state.prewarm.forget_before(hour_of(now));
    let mut complete = true;
    for scope in state.prewarm.due(Instant::now()) {
        let key = scope.key();
        let lease = tokio::task::spawn_blocking(move || share_prewarm(&key)).await;
        let _guard = match lease {
            Ok(Share::Mine(guard)) => guard,
            Ok(Share::Theirs) => continue,
            Err(e) => {
                tracing::warn!("the prewarm lease panicked: {e}");
                complete = false;
                continue;
            }
        };
        for at_ms in warm_times(now) {
            let (sections, sharing) = run_of(state, &scope, at_ms);
            if sections.is_empty() {
                continue;
            }
            let digest = digest(&sections, &sharing);
            if state.prewarm.is_sealed(&scope, hour_of(at_ms), digest) {
                continue;
            }
            let started = Instant::now();
            match seal_run(state, &sections, &sharing).await {
                Ok(streams) => {
                    tracing::info!(
                        scope = %scope.key(),
                        hour = hour_of(at_ms),
                        sections = sections.len(),
                        took_ms = started.elapsed().as_millis() as u64,
                        "shared journal run sealed ahead"
                    );
                    state
                        .prewarm
                        .record(scope.clone(), hour_of(at_ms), Sealed { digest, streams });
                }
                Err(e) => {
                    tracing::warn!(scope = %scope.key(), "a shared journal run did not seal: {e}");
                    complete = false;
                }
            }
        }
    }
    // Only a pass that knows every current run may say what none holds. On
    // the blocking pool: its lease waits for the vote.
    if complete {
        let sweeping = Arc::clone(state);
        if let Err(e) = tokio::task::spawn_blocking(move || sweep(&sweeping)).await {
            tracing::warn!("the shared journal sweep panicked: {e}");
        }
    }
}

/// Seal `sections` as a new conversation would, wait until they are durable,
/// and let go of them. The streams they sealed under.
async fn seal_run(
    state: &Arc<InferenceState>,
    sections: &[Section],
    sharing: &[Sharing],
) -> anyhow::Result<Vec<StreamId>> {
    let forking = Arc::clone(state);
    let mut seq: Sequence =
        tokio::task::spawn_blocking(move || forking.base_conv.fork_at(DecodePriority::Low))
            .await?
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    let timeline = seq.timeline_id();
    state.engine.mark_timeline_transient(timeline);
    let wants: Vec<Want<'_>> = sections
        .iter()
        .zip(sharing)
        .map(|(s, &sharing)| Want {
            name: &s.name,
            text: &s.text,
            sharing,
        })
        .collect();
    let mut carried = Carried::new(true);
    let sealed = carried
        .reconcile(
            &mut seq,
            JOURNAL,
            &wants,
            Linking::Chain,
            &[],
            Some(SEAL_WAIT),
            tokio::time::sleep,
        )
        .await;
    let substrate = state.engine.conversation();
    let streams: Vec<(StreamId, bool)> = seq
        .submitted_sections()
        .iter()
        .zip(sharing)
        .filter_map(|(s, sharing)| {
            substrate
                .section_stream(s.section_id())
                .map(|stream| (stream, sharing.keeps_end_state()))
        })
        .collect();
    let all_sealed = sealed.len() == wants.len();
    if all_sealed {
        let deadline = Instant::now() + DURABLE_WAIT;
        loop {
            let durable = streams.iter().all(|&(stream, keeps)| {
                substrate.section_stream_is_persisted(stream)
                    && (!keeps
                        || substrate
                            .read_section_end_state(stream)
                            .is_ok_and(|kept| kept.is_some()))
            });
            if durable || Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    for want in &wants {
        seq.remove_section_named(want.name);
    }
    let tombstoning = Arc::clone(state);
    let tombstoned =
        tokio::task::spawn_blocking(move || tombstoning.engine.tombstone_timeline(timeline))
            .await?;
    if let Err(e) = tombstoned {
        tracing::warn!(
            timeline = timeline.raw(),
            "the warmer's conversation was not tombstoned: {e}"
        );
    }
    drop(seq);
    if !all_sealed {
        anyhow::bail!("{} of {} sections sealed", sealed.len(), wants.len());
    }
    Ok(streams.into_iter().map(|(s, _)| s).collect())
}

/// Retire the shared journal streams no current run holds and no conversation
/// of this member reads (`docs/shared_journal_sections.md` §6). Ended as a
/// rebuild is: a conversation of another member still reading one keeps its
/// resident copy, and one that comes back to it misses and seals it again.
fn sweep(state: &InferenceState) {
    let Share::Mine(_guard) = share_prewarm("sweep") else {
        return;
    };
    let substrate = state.engine.conversation();
    let (shared, held) = {
        let read = substrate.read();
        (read.shared_section_streams(), read.held_section_streams())
    };
    let gone = retirable(&shared, &state.prewarm.current(), &held);
    for &stream in &gone {
        if let Err(e) =
            substrate.retire_section_generation(stream, "shared journal section superseded")
        {
            tracing::warn!(stream = stream.0, "a shared section was not retired: {e}");
        }
    }
    if !gone.is_empty() {
        tracing::info!(
            retired = gone.len(),
            "retired superseded shared journal sections"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_conversation_is_in_its_owners_scope_or_the_threadless_one() {
        assert_eq!(
            Scope::of("mcp:abc:~chat/zen"),
            Scope::Owner("abc".to_string())
        );
        assert_eq!(Scope::of("c-123"), Scope::Threadless);
        assert_eq!(Scope::Owner("abc".into()).conv_id(), "mcp:abc:~prewarm/zen");
        assert_eq!(Scope::Owner("abc".into()).key(), "owner/abc");
        assert_eq!(Scope::Threadless.key(), "threadless");
    }

    /// The next hour's run is sealed in the hour's last five minutes, and not
    /// before.
    #[test]
    fn the_next_hours_run_is_dated_within_the_lead() {
        let hour = 490_000 * HOUR_MS;
        assert_eq!(warm_times(hour + 10 * 60_000), vec![hour + 10 * 60_000]);
        assert_eq!(
            warm_times(hour + HOUR_MS - HOUR_LEAD_MS),
            vec![hour + HOUR_MS - HOUR_LEAD_MS, hour + HOUR_MS]
        );
        assert_eq!(
            warm_times(hour + HOUR_MS - 1),
            vec![hour + HOUR_MS - 1, hour + HOUR_MS]
        );
    }

    /// A scope is warm for a day after its last conversation, the threadless one
    /// always; a run once sealed is not sealed again until it moves, and an
    /// hour gone takes its streams out of the current set.
    #[test]
    fn scopes_age_out_and_sealed_runs_are_remembered_by_hour() {
        let p = Prewarm::default();
        let t0 = Instant::now();
        p.note("mcp:abc:~one/zen", t0);
        p.note("mcp:abc:~two/zen", t0);
        assert_eq!(
            p.due(t0),
            vec![Scope::Owner("abc".into()), Scope::Threadless]
        );
        assert_eq!(p.due(t0 + SCOPE_TTL), vec![Scope::Threadless]);

        let owner = Scope::Owner("abc".into());
        p.record(
            owner.clone(),
            7,
            Sealed {
                digest: 1,
                streams: vec![StreamId(10), StreamId(11)],
            },
        );
        p.record(
            owner.clone(),
            8,
            Sealed {
                digest: 2,
                streams: vec![StreamId(12)],
            },
        );
        assert!(p.is_sealed(&owner, 7, 1));
        assert!(
            !p.is_sealed(&owner, 7, 2),
            "a run that moved is sealed again"
        );
        assert_eq!(p.current().len(), 3);
        p.forget_before(8);
        assert_eq!(p.current(), HashSet::from([StreamId(12)]));
    }

    #[test]
    fn the_sweep_retires_only_what_nothing_holds() {
        let shared = [StreamId(1), StreamId(2), StreamId(3), StreamId(4)];
        let current = HashSet::from([StreamId(1)]);
        let held = HashSet::from([StreamId(3)]);
        assert_eq!(
            retirable(&shared, &current, &held),
            vec![StreamId(2), StreamId(4)]
        );
    }
}
