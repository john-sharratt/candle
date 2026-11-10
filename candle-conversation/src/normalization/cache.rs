//! The runtime cache of per-scope normalization state.
//!
//! In-memory and **not persisted** — it is derived from query traffic and rebuilt
//! empty each process load (see the module docs). `normalize` is the read path
//! (every reprojection); `observe` is the sole writer (once per turn at seal, and
//! the warm-ups).
//!
//! Shared by every thread that scores, and held by none: each scope's state is
//! one immutable value behind an atomic cell. A read loads it and normalizes
//! against exactly what it loaded; an observe folds into a copy and swaps it
//! in, folding again on the newer state when another observe swapped first.

use std::sync::Arc;

use arc_swap::ArcSwap;

use super::scope::ScopeState;
use super::{ChildKey, NormConfig, ScopeKey};

/// Per-scope background normalization, keyed by [`ScopeKey`]. Starts empty;
/// each scope fills in lazily as it is first observed.
#[derive(Default)]
pub struct NormalizationCache {
    cfg: NormConfig,
    scopes: papaya::HashMap<ScopeKey, ArcSwap<ScopeState>>,
}

impl NormalizationCache {
    pub fn new(cfg: NormConfig) -> Self {
        NormalizationCache {
            cfg,
            scopes: papaya::HashMap::new(),
        }
    }

    /// `scope`'s state as it stands, if it has ever been observed.
    fn state(&self, scope: &ScopeKey) -> Option<Arc<ScopeState>> {
        self.scopes.pin().get(scope).map(|s| s.load_full())
    }

    pub fn config(&self) -> &NormConfig {
        &self.cfg
    }

    /// Read path: each child's raw score on the band, as tail surprise against
    /// its background in `scope` ([`NormConfig::full_hit_tail`]). A scope
    /// never observed reads every child against the probe's own scores. Pure.
    ///
    /// `probe_tokens` is the length of the probe `raw` was scored with, in
    /// signature tokens. It is required rather than optional because a raw score
    /// is a sum over the probe and the band is a rate — see
    /// [`NormConfig::probe_t_ref`].
    ///
    /// Normalize a competition's children in ONE call: a child too new to have
    /// a background is read against the others, which a call naming it alone
    /// cannot show it.
    pub fn normalize(
        &self,
        scope: &ScopeKey,
        raw: &[(ChildKey, f32)],
        probe_tokens: usize,
    ) -> Vec<(ChildKey, f32)> {
        match self.state(scope) {
            Some(s) => s.normalize(raw, &self.cfg, probe_tokens),
            None => ScopeState::default().normalize(raw, &self.cfg, probe_tokens),
        }
    }

    /// Write path: fold one probe's raw scores into `scope`'s backgrounds,
    /// creating the scope on first observation — every child's but `answer`'s,
    /// when the probe is known to be about one of them (a warm-up exemplar, a
    /// file's own turn). A live seal knows no answer and teaches every child:
    /// a hub that scores on everything then carries those scores as its own
    /// background, which is what mutes it.
    ///
    /// Every `(group, timeline)` scope is retained independently. A belief group
    /// like `code_reading` has MANY simultaneously-active timelines — one per
    /// ingested file — and each keeps its own backgrounds. (An earlier version
    /// evicted a group's *other* timelines on every observe, which wiped every
    /// file but the last and let a promiscuous file win every query.) Stale
    /// scopes from a repo_map re-scan are left in place — dead once their
    /// timeline is inactive, bounded by the re-scan count.
    ///
    /// **Idempotent per observation.** `source` identifies the evidence — the
    /// turn this scoring pass came from — and a scope folds each source exactly
    /// once. Re-observing it is a no-op, which is what lets the backgrounds be
    /// rebuilt on EVERY load, from empty, against a substrate already on disk,
    /// while the same scopes keep learning from live traffic afterwards.
    /// Deduplicating on the SCORE instead would be wrong: a promiscuous child
    /// scores about the same on everything, so it would be recorded once and
    /// never learn that it is loud across all traffic.
    ///
    /// Fast on the hot path: one hash lookup rejects an already-folded source
    /// before touching any child. `probe_tokens` is the length of the probe
    /// `raw` was scored with; see [`NormConfig::probe_t_ref`].
    pub fn observe(
        &self,
        scope: &ScopeKey,
        source: u64,
        raw: &[(ChildKey, f32)],
        answer: Option<&ChildKey>,
        probe_tokens: usize,
    ) {
        let scopes = self.scopes.pin();
        let cell = scopes.get_or_insert_with(scope.clone(), || {
            ArcSwap::from_pointee(ScopeState::default())
        });
        cell.rcu(|current| {
            let mut next = ScopeState::clone(current);
            if next.mark_observed(source) {
                next.observe(raw, answer, &self.cfg, probe_tokens);
            }
            next
        });
    }

    #[cfg(test)]
    pub(super) fn background_of(&self, scope: &ScopeKey, child: &ChildKey) -> Option<u32> {
        self.state(scope).and_then(|s| s.background_of(child))
    }
}
