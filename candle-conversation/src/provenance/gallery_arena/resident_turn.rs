//! Holds one turn's pages on the card, and the bookkeeping that says what they
//! serve, who holds them and how often they are scanned.
//!
//! Dropping a turn's page run returns every slot to its arena, so eviction is just
//! dropping the turn. A turn carries the run id that changes whenever its addresses
//! do. A pin taken by a scan keeps eviction from freeing pages a launch is still
//! reading. The interval between a turn's scans is a weighted average, which lets
//! the keep trade count how often keeping the turn resident saves a re-upload.

use super::*;

/// A turn's owned run of page slots, in page order. Dropping it returns every slot
/// to the gallery's arenas — so evicting a turn is just dropping its
/// [`ResidentTurn`].
pub(super) struct PageRun {
    pub(super) slots: Vec<ArenaSlot>,
}

impl PageRun {
    /// Device addresses of the run's pages, in page order.
    pub(super) fn addrs(&self) -> Vec<u64> {
        self.slots.iter().map(ArenaSlot::ptr).collect()
    }
}

pub(super) struct ResidentTurn {
    pub(super) fingerprint: u64,
    pub(super) run: PageRun,
    /// Identifies `run`'s current page addresses: fresh on every upload and
    /// every compaction move. A cached scan index is valid exactly while every
    /// turn it references still carries the run id it was built against.
    pub(super) run_id: u64,
    /// Which of the turn's tokens its pages hold, packed in order, and the
    /// keep they were chosen under (`docs/gallery_dedup.md`).
    pub(super) mask: Arc<KeepMask>,
    pub(super) keep: Keep,
    pub(super) lru: u64,
    /// When a scan last used it.
    pub(super) used: Instant,
    /// How long between the scans that use it, averaged — with `used`, how
    /// often keeping it resident spares a re-upload (see [`eviction`]). `None`
    /// until a second scan has used it.
    pub(super) interval: Option<Duration>,
    /// Non-zero while a scan is reading this turn's pages — the governor's
    /// eviction skips pinned turns so it can never free a page an in-flight launch
    /// dereferences (a scan pins every turn it touches, then unpins after launch).
    pub(super) pinned: u32,
    /// Who it is held for, widened by every read ([`Holder::widen`]).
    pub(super) holder: Holder,
    /// The fold groups its pages hold — the union of what its reads weigh
    /// (`docs/gallery_residency.md` §4).
    pub(super) groups: GroupSet,
}

impl ResidentTurn {
    /// Whether its pages serve a read of the content `fingerprint` names,
    /// weighing `groups` of the tokens `keep` keeps.
    pub(super) fn serves(&self, fingerprint: u64, groups: GroupSet, keep: Keep) -> bool {
        self.fingerprint == fingerprint && self.groups.covers(groups) && self.keep.covers(keep)
    }

    /// A read found it serving: it is used now, at `lru`, held for `holder`
    /// too, and pinned for a scan when `pin`.
    pub(super) fn hit(&mut self, lru: u64, holder: Holder, pin: bool) -> Ensured {
        self.lru = lru;
        self.touch(Instant::now());
        self.holder = self.holder.widen(holder);
        if pin {
            self.pinned += 1;
        }
        Ensured {
            addrs: self.run.addrs(),
            run_id: self.run_id,
            groups: self.groups,
            mask: Arc::clone(&self.mask),
            uploaded: false,
        }
    }

    /// A scan used it at `now`: its interval takes in the time since the last.
    pub(super) fn touch(&mut self, now: Instant) {
        let since = now.saturating_duration_since(self.used);
        self.interval = Some(match self.interval {
            Some(i) => i.mul_f64(1.0 - INTERVAL_WEIGHT) + since.mul_f64(INTERVAL_WEIGHT),
            None => since,
        });
        self.used = now;
    }
}

/// Weight of the newest gap between scans in a turn's interval.
pub(super) const INTERVAL_WEIGHT: f64 = 0.5;
