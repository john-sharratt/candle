//! The criticality-ranked relief ladder: registration, the escalating relief
//! loop, and the evictable-memory estimate the forecast uses.
//!
//! Callers register a closure per (`AllocClass`, `Criticality`) for the resource
//! they own (KV wraps its release/compact/evict/compress/drop fns; the expert
//! cache wraps slot→pinned and pool-shrink). Relief runs cheapest-first and only
//! synchronises the GPU at the `Critical` rung — see
//! `docs/vram_governor_design.md` §8.

use std::sync::atomic::Ordering;
use std::time::Instant;

use super::{AllocClass, Criticality, VramGovernor};
use crate::Result;

/// What the governor asks a relief closure to free.
#[derive(Clone, Copy, Debug)]
pub struct ReliefRequest {
    /// Bytes the governor would like this closure to free (best-effort).
    pub want: u64,
    /// The rung this closure is registered at (so one closure can branch on it).
    pub tier: Criticality,
}

/// What a relief closure reports back. `freed_est` is the bytes it *queued* to
/// free (async); the next measurement reflects the physical reclamation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReliefOutcome {
    pub freed_est: u64,
}

impl ReliefOutcome {
    pub fn new(freed_est: u64) -> Self {
        Self { freed_est }
    }
}

/// Outcome of an escalating relief pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReliefResult {
    /// The target headroom was reached; carries total bytes freed (estimate).
    Relieved(u64),
    /// The whole ladder ran and the target was still not met (circuit breaker
    /// tripped — the caller surfaces an OOM instead of spinning).
    Exhausted(u64),
}

impl ReliefResult {
    pub fn freed(&self) -> u64 {
        match self {
            ReliefResult::Relieved(b) | ReliefResult::Exhausted(b) => *b,
        }
    }
    pub fn is_relieved(&self) -> bool {
        matches!(self, ReliefResult::Relieved(_))
    }
}

/// A caller-provided relief executor for [`VramGovernor::relieve_with`]. The
/// governor drives it cheapest-rung-first; the implementor (e.g. the scheduler)
/// borrows its own state and runs the actual mechanism for `tier`, returning the
/// bytes it freed (an estimate — the governor re-measures reality between rungs).
///
/// `tier` maps to: `Trivial` = release empty arenas + trim pool; `Cheap` =
/// lossless compaction; `Moderate` = compress-to-free (bring forward the
/// quantization of completed float turns — a net shrink that keeps them
/// resident, so it precedes eviction's move-to-RAM); `Costly` = hot→warm
/// eviction; `Critical` = drop-to-cold / expert-pool shrink. Return `0` for a
/// rung the caller doesn't service synchronously.
pub trait KvReliefDriver {
    fn relieve(&mut self, tier: Criticality, want: u64) -> u64;
}

type ReliefFn = Box<dyn Fn(ReliefRequest) -> ReliefOutcome + Send + Sync>;
type EvictFn = Box<dyn Fn() -> u64 + Send + Sync>;

struct ReliefEntry {
    id: u64,
    relief: ReliefFn,
    evictable: EvictFn,
}

/// The per-(class, tier) table of registered relievers.
#[derive(Default)]
pub(crate) struct ReliefRegistry {
    // [class][tier] -> registered entries.
    by_class_tier: [[Vec<ReliefEntry>; 5]; AllocClass::COUNT],
    next_id: u64,
}

/// Opaque token to later unregister a relief closure.
#[derive(Clone, Copy, Debug)]
pub struct ReliefHandle {
    class: AllocClass,
    tier: Criticality,
    id: u64,
}

impl VramGovernor {
    /// Register a relief closure for `class` at `tier`, plus an `evictable`
    /// reporter (bytes this owner could reversibly free at this tier) used by the
    /// forecast (§9.2). Returns a handle for [`Self::unregister_relief`].
    pub fn register_relief<R, E>(
        &self,
        class: AllocClass,
        tier: Criticality,
        relief: R,
        evictable: E,
    ) -> ReliefHandle
    where
        R: Fn(ReliefRequest) -> ReliefOutcome + Send + Sync + 'static,
        E: Fn() -> u64 + Send + Sync + 'static,
    {
        let mut reg = self.relief.write().unwrap();
        reg.next_id += 1;
        let id = reg.next_id;
        reg.by_class_tier[class.idx()][tier.idx()].push(ReliefEntry {
            id,
            relief: Box::new(relief),
            evictable: Box::new(evictable),
        });
        ReliefHandle { class, tier, id }
    }

    /// Remove a previously registered relief closure. Returns whether it was found.
    pub fn unregister_relief(&self, handle: ReliefHandle) -> bool {
        let mut reg = self.relief.write().unwrap();
        let slot = &mut reg.by_class_tier[handle.class.idx()][handle.tier.idx()];
        let before = slot.len();
        slot.retain(|e| e.id != handle.id);
        slot.len() != before
    }

    /// Bytes that registered relievers report they could reversibly free at rungs
    /// up to and including `up_to`, across all classes. Feeds the forecast.
    pub fn evictable_estimate(&self, up_to: Criticality) -> u64 {
        let reg = self.relief.read().unwrap();
        let mut total = 0u64;
        for class in 0..AllocClass::COUNT {
            for tier in Criticality::ALL {
                if tier > up_to {
                    continue;
                }
                for entry in &reg.by_class_tier[class][tier.idx()] {
                    total = total.saturating_add((entry.evictable)());
                }
            }
        }
        total
    }

    /// Run every closure registered for `class` at `tier`, threading the
    /// remaining shortfall as `want`. Returns total `freed_est`.
    fn invoke_tier(&self, class: AllocClass, tier: Criticality, want: u64) -> u64 {
        let reg = self.relief.read().unwrap();
        let mut freed = 0u64;
        for entry in &reg.by_class_tier[class.idx()][tier.idx()] {
            let remaining = want.saturating_sub(freed);
            let out = (entry.relief)(ReliefRequest {
                want: remaining,
                tier,
            });
            freed = freed.saturating_add(out.freed_est);
        }
        freed
    }

    /// Run exactly one rung: at `Critical`, sync the GPU before and after and
    /// pull in cross-class relievers (rate-limited); at lower rungs, just this
    /// class's relievers, no sync. Returns bytes freed (estimate). Shared by the
    /// escalating loop and the allocation-retry path.
    pub(crate) fn run_tier_with_sync(
        &self,
        trigger: AllocClass,
        tier: Criticality,
        want: u64,
    ) -> u64 {
        if tier == Criticality::Critical {
            if !self.try_enter_critical() {
                return 0;
            }
            self.do_sync();
            let mut freed = self.invoke_tier(trigger, tier, want);
            freed = freed.saturating_add(self.invoke_cross_class(
                trigger,
                tier,
                want.saturating_sub(freed),
            ));
            self.do_sync();
            freed
        } else {
            self.invoke_tier(trigger, tier, want)
        }
    }

    /// Run relief at `tier` for every class *except* `trigger` — used only at the
    /// `Critical` rung, where KV and experts finally negotiate.
    fn invoke_cross_class(&self, trigger: AllocClass, tier: Criticality, want: u64) -> u64 {
        let mut freed = 0u64;
        for class in AllocClass::ALL {
            if class == trigger {
                continue;
            }
            freed = freed.saturating_add(self.invoke_tier(class, tier, want.saturating_sub(freed)));
        }
        freed
    }

    /// Rate-limit entry into the `Critical` rung (the only rung that syncs the
    /// GPU). Returns `true` if this call may proceed.
    fn try_enter_critical(&self) -> bool {
        let interval = self.config().critical_min_interval_ms;
        if interval == 0 {
            return true;
        }
        let mut last = self.last_critical.lock().unwrap();
        let now = Instant::now();
        let allow = match *last {
            Some(prev) => prev.elapsed().as_millis() as u64 >= interval,
            None => true,
        };
        if allow {
            *last = Some(now);
        }
        allow
    }

    fn record_relief(&self, tier: Criticality, freed: u64) {
        *self.last_relief.lock().unwrap() = Some((tier, freed));
    }

    /// Proactive, threshold-gated relief: run each rung whose threshold the live
    /// headroom has crossed, cheapest-first, stopping as soon as a rung is not
    /// tripped (thresholds descend, so nothing deeper is tripped either). This is
    /// the path the scheduler/budget-watcher calls; it withholds KV eviction
    /// until headroom nears the floor.
    pub fn relieve_pressure(&self, trigger: AllocClass) -> Result<ReliefResult> {
        let healthy = self.tier_threshold(Criticality::Trivial);
        let mut freed = 0u64;
        let mut deepest = None;
        for tier in Criticality::ALL {
            let hr = self.available()?;
            let thr = self.tier_threshold(tier);
            if hr > thr {
                break; // this rung not tripped ⇒ nothing deeper is (thresholds descend)
            }
            if tier == Criticality::Critical {
                if !self.try_enter_critical() {
                    break;
                }
                self.do_sync();
                if self.available()? > thr {
                    break;
                }
            }
            let want = healthy.saturating_sub(self.available()?);
            freed = freed.saturating_add(self.invoke_tier(trigger, tier, want));
            if tier == Criticality::Critical {
                let want2 = healthy.saturating_sub(self.available()?);
                freed = freed.saturating_add(self.invoke_cross_class(trigger, tier, want2));
                self.do_sync();
            }
            deepest = Some(tier);
        }
        if let Some(tier) = deepest {
            self.record_relief(tier, freed);
        }
        let hr = self.available()?;
        Ok(if hr >= healthy {
            ReliefResult::Relieved(freed)
        } else {
            ReliefResult::Exhausted(freed)
        })
    }

    /// Escalating relief toward a target headroom: run rungs cheapest-first until
    /// live headroom reaches `target` or the ladder is exhausted. Unlike
    /// [`Self::relieve_pressure`] this ignores per-rung thresholds — the caller is
    /// demanding a specific amount of room and must climb until it has it.
    pub fn relieve_to(&self, trigger: AllocClass, target: u64) -> Result<ReliefResult> {
        let mut freed = 0u64;
        for tier in Criticality::ALL {
            if self.available()? >= target {
                self.record_relief(tier, freed);
                return Ok(ReliefResult::Relieved(freed));
            }
            if tier == Criticality::Critical {
                if !self.try_enter_critical() {
                    break;
                }
                self.do_sync();
                if self.available()? >= target {
                    return Ok(ReliefResult::Relieved(freed));
                }
            }
            let want = target.saturating_sub(self.available()?);
            freed = freed.saturating_add(self.invoke_tier(trigger, tier, want));
            if tier == Criticality::Critical {
                let want2 = target.saturating_sub(self.available()?);
                freed = freed.saturating_add(self.invoke_cross_class(trigger, tier, want2));
                self.do_sync();
            }
        }
        let hr = self.available()?;
        self.record_relief(Criticality::Critical, freed);
        Ok(if hr >= target {
            ReliefResult::Relieved(freed)
        } else {
            ReliefResult::Exhausted(freed)
        })
    }

    /// Escalating relief toward `target` headroom driven by a **borrowed** driver
    /// instead of pre-registered `'static` closures. The governor owns the
    /// policy — measure, cheapest-rung-first, escalate only while below `target`,
    /// GPU sync only at `Critical` — and calls `driver.relieve(tier, want)` for
    /// the mechanism at each rung, re-measuring between rungs.
    ///
    /// This is the production path: the caller (the scheduler) implements
    /// [`KvReliefDriver`] borrowing its own `&mut` state, so the eviction
    /// mechanisms run on the caller's thread with full working-set context — no
    /// `Arc`/roster plumbing, and no off-thread eviction racing a live forward.
    pub fn relieve_with(
        &self,
        target: u64,
        driver: &mut dyn KvReliefDriver,
    ) -> Result<ReliefResult> {
        let mut freed = 0u64;
        for tier in Criticality::ALL {
            if self.available()? >= target {
                self.record_relief(tier, freed);
                return Ok(ReliefResult::Relieved(freed));
            }
            if tier == Criticality::Critical {
                if !self.try_enter_critical() {
                    break;
                }
                self.do_sync();
                if self.available()? >= target {
                    return Ok(ReliefResult::Relieved(freed));
                }
            }
            let want = target.saturating_sub(self.available()?);
            freed = freed.saturating_add(driver.relieve(tier, want));
            if tier == Criticality::Critical {
                self.do_sync();
            }
        }
        let hr = self.available()?;
        self.record_relief(Criticality::Critical, freed);
        Ok(if hr >= target {
            ReliefResult::Relieved(freed)
        } else {
            ReliefResult::Exhausted(freed)
        })
    }

    /// The last relief episode `(deepest tier, bytes freed)`, for diagnostics.
    pub fn last_relief(&self) -> Option<(Criticality, u64)> {
        *self.last_relief.lock().unwrap()
    }

    /// Number of relief closures registered (diagnostics / tests).
    pub fn relief_count(&self) -> usize {
        let reg = self.relief.read().unwrap();
        let mut n = 0;
        for class in 0..AllocClass::COUNT {
            for tier in 0..5 {
                n += reg.by_class_tier[class][tier].len();
            }
        }
        n
    }

    /// How many times the `Critical`-rung GPU sync has fired since construction
    /// (diagnostics / tests).
    pub fn sync_count(&self) -> u64 {
        self.sync_calls.load(Ordering::Relaxed)
    }
}
