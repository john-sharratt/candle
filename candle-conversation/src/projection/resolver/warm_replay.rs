//! Replays a window of recent dialogue turns to warm the normalization backgrounds
//! at load.
//!
//! A restart would otherwise begin with cold backgrounds, because the cache is
//! runtime-only. Only untagged dialogue turns are replayed, up to a fixed cap taken
//! from the most recent by timeline and index, so which turns warm the levels does
//! not depend on map order. The replay re-arms after an ingest reconcile, so files
//! added later are learned too rather than dominating queries at an unnormalized
//! score.

use super::*;
use crate::substrate::Located;

/// Upper bound on how many recent dialogue turns the normalization warm-up
/// replays on load (`take_warm_replay`): 512 samples of every child's background
/// read its tail to the 1-in-100 a full hit is set at, and the cap bounds the
/// one-time cost.
pub(super) const WARM_REPLAY_MAX_TURNS: usize = 512;

/// One sealed dialogue turn the normalization warm-up replays — see
/// [`Conversation::take_warm_replay`].
pub struct WarmProbe {
    /// The turn's stream id: the observation's source, so a scope folds it once.
    pub source: u64,
    /// The turn's stored wide-Q signature.
    pub probe: Vec<WideQSig>,
}

impl Conversation {
    /// The dialogue replay that warms the normalization backgrounds — the
    /// substrate's existing **sealed dialogue turns**, so a freshly-loaded
    /// process starts WARM (the cache is runtime-only, so without this a restart
    /// would be cold until new traffic accrued). See
    /// `docs/provenance_score_normalization.md` §4.4.
    ///
    /// Returns the probes when the replay is armed and disarms it; `None`
    /// otherwise. The caller scores each with [`Self::score_warm_probe`] — the
    /// scheduler, on its gallery arena, in the gaps higher-priority work leaves.
    /// Early reprojections may read a still-warming cache: that is best-effort
    /// by construction — the backgrounds are runtime-derived and the
    /// once-per-turn seal-observe keeps warming them — so only how quickly
    /// generic candidates get discounted depends on it.
    ///
    /// The last [`WARM_REPLAY_MAX_TURNS`] dialogue turns, ordered by
    /// `(timeline, turn index)` so which turns are replayed does not depend on
    /// `HashMap` iteration order, and bounded to the recent window. Only
    /// dialogue turns (empty-tagged — gallery turns are tagged) are probes; the
    /// still-decoding current turn has no sealed signature yet.
    pub fn take_warm_replay(&self, target: ProjectionTarget) -> Option<Vec<WarmProbe>> {
        // Never armed from an ingest-time (append-only target) scoring pass:
        // during corpus load the corpus is still moving, and a replay then warms
        // backgrounds the next file retires. A real query projects against a
        // non-append-only (dialogue) target, by which point it is stable.
        if self.inner.read().is_append_only_layer(target.layer) {
            return None;
        }
        // Fire once, but RE-ARMABLE: `reset_normalization_warm` clears the flag after
        // an ingest reconcile so the next projection re-learns the newly-ingested
        // files' backgrounds. Firing only once would warm only the corpus present at
        // the first query — files the background reconcile adds afterwards would
        // stay cold and dominate every query at an un-normalized score.
        if self.normalization_warm.swap(true, AtomicOrdering::SeqCst) {
            return None;
        }
        // Ordered first and read after: only the turns the replay keeps are read
        // back and decoded, not every dialogue turn in the store.
        let sub = self.inner.read();
        let mut turns: Vec<(u64, u32, Located, RecordIdentity)> = sub
            .dialogue_turn_streams()
            .filter_map(|turn| {
                let value = turn.runtime.wide_q_sigs.clone()?;
                let id = turn.runtime.ids.wide_q_sigs;
                Some((turn.decl.timeline_id, turn.index.0, value, id))
            })
            .collect();
        turns.sort_by_key(|(tl, idx, _, _)| (*tl, *idx));
        let start = turns.len().saturating_sub(WARM_REPLAY_MAX_TURNS);
        let replay: Vec<WarmProbe> = turns
            .drain(start..)
            .filter_map(|(tl, idx, value, id)| {
                let bytes = sub.located_bytes(&value, id).ok()?;
                let probe = decode_wide_sigs_for_scoring(&bytes)?;
                (!probe.is_empty()).then(|| WarmProbe {
                    source: turn_stream_id(tl, idx).0,
                    probe,
                })
            })
            .collect();
        drop(sub);
        self.warm_replay_progress.add(replay.len());
        Some(replay)
    }

    /// The target a replay queued at load scores for: the timeline of the
    /// last turn the replay would take — the last sealed dialogue turn by
    /// `(timeline, turn index)`, as [`Self::take_warm_replay`] orders them —
    /// with that timeline's layer and group. `None` when no dialogue turn
    /// carries a signature, or its timeline is unregistered or on an
    /// append-only layer (which never arms the replay).
    pub fn warm_replay_target(&self) -> Option<ProjectionTarget> {
        let sub = self.inner.read();
        let (tl, _) = sub
            .dialogue_turn_streams()
            .filter(|turn| turn.runtime.wide_q_sigs.is_some())
            .map(|turn| (turn.decl.timeline_id, turn.index.0))
            .max()?;
        let timeline = TimelineId::from_raw(tl)?;
        let (layer, group) = sub.timeline_target(timeline)?;
        if sub.is_append_only_layer(layer) {
            return None;
        }
        Some(ProjectionTarget {
            layer,
            group,
            timeline,
        })
    }

    /// `(scored, total)` of the replay now draining — `None` once every turn
    /// it handed out has been scored.
    pub fn warm_replay_progress(&self) -> Option<(usize, usize)> {
        self.warm_replay_progress.snapshot()
    }

    /// `n` handed-out replay turns that will never be scored: the scheduler
    /// dropped them when this substrate detached.
    pub fn warm_replay_dropped(&self, n: usize) {
        self.warm_replay_progress.dropped(n);
    }

    /// Re-arm the normalization warm-up (see [`Self::take_warm_replay`]) so the
    /// next projection re-learns per-file backgrounds. Called after an ingest
    /// reconcile mints fresh file timelines the prior warm never scanned.
    pub fn reset_normalization_warm(&self) {
        self.normalization_warm.store(false, AtomicOrdering::SeqCst);
    }

    /// Fold one replayed dialogue turn into the backgrounds — a seal-style
    /// observe against every belief node, whose normalized scores are
    /// discarded.
    ///
    /// These probes are the UNTAGGED dialogue turns, so they teach the unscoped
    /// turn groups and — by the routing in `Observe::teaches` — no tag-scoped
    /// collection. Both scans are asked for; the lens rule decides which scopes
    /// actually learn, rather than the call site guessing. That is what keeps
    /// dialogue from degrading the tool lens while leaving both free to learn
    /// from their own traffic. Keyed on the turn itself, so a turn a scope has
    /// already folded changes nothing there.
    pub fn score_warm_probe(
        &self,
        schema: &Schema,
        target: ProjectionTarget,
        warm: &WarmProbe,
        arena: Option<&GalleryArena>,
    ) {
        let observe = Observe::Yes {
            tags: &[],
            source: warm.source,
        };
        let arena = arena.map(|a| a as &dyn BeliefArena);
        let _ =
            self.score_belief_collections(&schema.system_prompt, &warm.probe, None, observe, arena);
        let mut throwaway = ProjectionScores::new();
        for layer in &schema.layers {
            let _ = self.score_belief_groups(
                layer,
                target,
                &warm.probe,
                None,
                &mut throwaway,
                observe,
                arena,
            );
        }
        self.warm_replay_progress.scored_one();
    }
}
