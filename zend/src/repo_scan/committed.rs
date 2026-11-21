//! The durable record of which `repo_map` folders have been ingested, and the
//! boot sweep that retires the attempts a crash left behind.
//!
//! A folder's conversation is committed only when it carries both its directory
//! tag and its content key. The directory tag is written at creation and the key
//! only after the ingest succeeds, so a directory with no key is an attempt that
//! never finished, whether it crashed or is another worker's in flight. The pass
//! plans from the keys read off the substrate, so a unit whose ingest failed is
//! queued again. A crashed partial still competes in provenance retrieval, so the
//! sweep retires it once at boot, before any pool runs, rather than during an
//! ingest, where an in-flight unit cannot be told from a crashed one.

use super::*;

/// Retire every crashed-partial `repo_map` conversation, up front.
///
/// A partial carries [`DIR_KEY`] but no [`CONTENT_KEY`] — the directory tag is
/// written at conversation creation and the content key only after the unit's
/// ingest succeeds, so the pair says "this attempt started and never finished".
/// A conversation ingested before units were keyed by content carries no
/// content key either, and goes the same way: nothing can scope it to a
/// conversation's files.
///
/// **Why here.** An attempt that fails retires its own conversation
/// ([`process_one_dir`]); what a crash left has nobody to retire it, and a
/// pool cannot tell it from another worker's attempt in flight — so it is
/// retired once, at boot, before any pool runs, or it simply stays live.
///
/// Live is the problem. A partial is a half-built chain: a request and a
/// `file_list` round-trip with the summary decode missing, and because it was
/// never coupled (the ingest writes its `TurnCoupling`s only after the whole
/// chain returns) nothing marks its halves as belonging together either. It
/// still carries `WideQSig`s, so it still competes in the provenance gather —
/// and a file listing is about the most promiscuous thing that can: dozens of
/// paths, matching almost any probe. Measured on one substrate after a hard kill
/// mid-ingest: 35 such chains, and the dialogue's projection selecting their
/// scaffolding turns — a `<tool_response>` listing paired with a `file_read`
/// call, from two unrelated folders, with the summary nowhere and the last call
/// answered by nothing.
///
/// Called ONCE per boot from the session's ingest pre-loop, for every layer the
/// operator did not `--disable-layer`.
///
/// Never called from [`ingest_units`]. A pass can overlap a pool that is
/// mid-flight, and an in-flight unit is indistinguishable from a crashed one by
/// metadata alone — both carry `dir` with no key — so sweeping there would
/// tombstone the conversation a worker is still building. The pre-loop runs
/// before the background ingest worker is even woken, so nothing of this
/// member's is in flight — but a peer's readings are, and they replicate here
/// begun and unkeyed, so only this member's partials, or a peer's past
/// [`PEER_GRACE`](crate::branch_ingest::reader::PEER_GRACE), are retired
/// ([`settles_here`]).
pub(crate) fn retire_crashed_partials(engine: &ConversationEngine) {
    let e = engine;
    let mut retired = 0usize;
    for (tl, dir) in e.conversations_with_metadata_key(DIR_KEY) {
        if ingest_committed(|key| e.has_conversation_metadata_key(tl, key)) || !settles_here(e, tl)
        {
            continue;
        }
        match e.tombstone_timeline(tl) {
            Ok(()) => retired += 1,
            Err(err) => tracing::warn!(
                target: "zend::repo_scan",
                dir = %dir,
                "tombstone of crashed-partial conversation failed: {err:#}",
            ),
        }
    }
    if retired > 0 {
        tracing::info!(
            target: "zend::repo_scan",
            retired,
            "retired repo_map conversations with no content key (crashed partials, \
             or ingested before units were keyed by content) so they leave the \
             provenance gather",
        );
    }
}

/// Tombstone EVERY `repo_map` conversation, committed or not — `--wipe-layer
/// repo_map`'s targeted counterpart to [`retire_crashed_partials`], which only
/// removes the never-committed half.
///
/// Called from the session's ingest pre-loop, before the background ingest
/// worker's first pass — which then finds nothing committed and ingests every
/// folder as new, exactly as it would on a truly fresh install. This is the
/// "test the background ingest end to end without deleting the whole
/// substrate" tool: unlike `--wipe-substrate`, every other layer's content
/// (dialogue, `code_reading` unless it too is named) survives untouched.
pub(crate) fn wipe_layer(engine: &ConversationEngine) {
    let e = engine;
    let mut wiped = 0usize;
    for (tl, dir) in e.conversations_with_metadata_key(DIR_KEY) {
        match e.tombstone_timeline(tl) {
            Ok(()) => wiped += 1,
            Err(err) => tracing::warn!(
                target: "zend::repo_scan",
                dir = %dir,
                "--wipe-layer repo_map: tombstone failed: {err:#}",
            ),
        }
    }
    tracing::info!(
        target: "zend::repo_scan",
        wiped,
        "--wipe-layer repo_map: every repo_map conversation tombstoned — the next \
         pass re-ingests the whole layer",
    );
}

/// Whether a `repo_map` conversation's ingest ever committed.
///
/// The two keys are a completion protocol, not two independent tags: [`DIR_KEY`]
/// is written at conversation creation and [`CONTENT_KEY`] only after the
/// unit's ingest succeeds, so their combination is the only durable record of
/// whether an attempt finished. `dir` alone therefore means "started, never
/// committed" — a crashed partial, or another worker's attempt in flight — and
/// that is what the boot sweep ([`retire_crashed_partials`]) keys on.
///
/// Named and separate so the rule is stated once and testable without an
/// engine; the sweep that applies it needs a loaded model and so is covered by
/// the engine-backed suite rather than here.
pub(super) fn ingest_committed(holds: impl Fn(&str) -> bool) -> bool {
    holds(CONTENT_KEY)
}

/// Metadata key holding a unit's directory. Distinct from `code_read`'s
/// `path` so the two layers' conversations are never taken for each other's.
pub(crate) const DIR_KEY: &str = "dir";

/// Every committed `repo_map` conversation: its content key and its folder —
/// what the pass plans against. Read from the substrate, never from what a
/// pass attempted: a unit whose ingest failed wrote no key, so it is not
/// committed and is queued again.
pub fn committed(engine: &ConversationEngine) -> Vec<Committed> {
    let e = engine;
    let keys: HashMap<TimelineId, String> = e
        .conversations_with_metadata_key(CONTENT_KEY)
        .into_iter()
        .collect();
    let quants = crate::branch_ingest::keys::quant_tags(e);
    let conv = e.conversation();
    let substrate = conv.read();
    e.conversations_with_metadata_key(DIR_KEY)
        .into_iter()
        .filter_map(|(timeline, dir)| {
            Some(Committed {
                timeline,
                key: keys.get(&timeline)?.clone(),
                subject: dir,
                stamp: substrate
                    .first_turn_stamp(timeline.raw())
                    .unwrap_or_default(),
                quant: quants.get(&timeline).copied(),
            })
        })
        .collect()
}
