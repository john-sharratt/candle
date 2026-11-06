//! One warm-tier budget for every substrate an engine holds.
//!
//! The warm budget (`candle::vram::host_ram_budget`) is what the machine's RAM
//! affords the warm KV tier as a whole, and an engine runs one persistence
//! thread per attached substrate (`docs/shared_model_host.md` §3.2), each
//! purging its own store. Each purging to the whole budget lets the tier grow
//! to the budget times the number of substrates — measured on a member carrying
//! zend and npcd: 8.0 GB warm against a 4 GiB budget, which the scheduler's
//! admission throttle, summing every substrate's warm bytes against the one
//! budget, then read as permanently over budget.
//!
//! So each thread holds a [`WarmShare`] of one [`WarmLedger`], stamps its own
//! warm bytes into it every pass ([`WarmShare::hold`]), and purges to
//! [`WarmShare::target`]: its
//! share of the budget in proportion to what it holds. While the substrates
//! together fit, every target is at least what its substrate holds and nothing
//! is purged; once they do not, each sheds in proportion, and together they
//! come to the budget.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The warm bytes every substrate of one engine holds, one entry per
/// [`WarmShare`]. A share dropped with its thread leaves a dead entry, pruned
/// at the next join.
#[derive(Default)]
pub struct WarmLedger {
    shares: Mutex<Vec<Weak<AtomicU64>>>,
}

/// One substrate's entry in a [`WarmLedger`].
pub struct WarmShare {
    own: Arc<AtomicU64>,
    ledger: Arc<WarmLedger>,
}

impl WarmLedger {
    /// A ledger with no substrates.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A share for one more substrate, holding nothing yet.
    pub fn join(self: &Arc<Self>) -> WarmShare {
        let own = Arc::new(AtomicU64::new(0));
        let mut shares = self.shares.lock().unwrap_or_else(|e| e.into_inner());
        shares.retain(|s| s.strong_count() > 0);
        shares.push(Arc::downgrade(&own));
        WarmShare {
            own,
            ledger: Arc::clone(self),
        }
    }

    /// Warm bytes across every live share.
    fn total(&self) -> u64 {
        self.shares
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(Weak::upgrade)
            .map(|s| s.load(Ordering::Relaxed))
            .fold(0u64, u64::saturating_add)
    }
}

impl WarmShare {
    /// Record that this substrate holds `own` warm bytes. Stamped every pass,
    /// an empty tier included, so a substrate that drained does not go on
    /// squeezing the others with what it held.
    pub fn hold(&self, own: u64) {
        self.own.store(own, Ordering::Relaxed);
    }

    /// What this substrate may keep of `budget`, the budget of every substrate
    /// together, given what each last [held](Self::hold).
    pub fn target(&self, budget: u64) -> u64 {
        share_of(
            self.own.load(Ordering::Relaxed),
            self.ledger.total(),
            budget,
        )
    }
}

/// What a substrate holding `own` of `total` warm bytes may keep of `budget`:
/// `budget` in proportion to `own`. At least `own` while `total` fits in
/// `budget`, so nothing is purged until the substrates together pass it.
fn share_of(own: u64, total: u64, budget: u64) -> u64 {
    if total <= budget {
        return budget;
    }
    (u128::from(budget) * u128::from(own) / u128::from(total)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// **Under the budget nothing is purged**: the target is the whole budget,
    /// at least what the substrate holds.
    #[test]
    fn substrates_that_fit_together_keep_everything() {
        assert_eq!(share_of(GIB, 3 * GIB, 4 * GIB), 4 * GIB);
        assert_eq!(share_of(0, 0, 4 * GIB), 4 * GIB);
    }

    /// **Over it, each sheds in proportion**, and the targets sum to the
    /// budget: 6 GiB and 2 GiB against 4 GiB keep 3 and 1.
    #[test]
    fn substrates_over_the_budget_share_it_in_proportion() {
        let budget = 4 * GIB;
        let (a, b) = (6 * GIB, 2 * GIB);
        let total = a + b;
        assert_eq!(share_of(a, total, budget), 3 * GIB);
        assert_eq!(share_of(b, total, budget), GIB);
    }

    /// **A substrate holding nothing is given nothing** once the others are
    /// over the budget — it has nothing to purge either way.
    #[test]
    fn an_empty_substrate_is_given_nothing_over_the_budget() {
        assert_eq!(share_of(0, 8 * GIB, 4 * GIB), 0);
    }

    /// **A zero budget purges everything.**
    #[test]
    fn a_zero_budget_keeps_nothing() {
        assert_eq!(share_of(GIB, GIB, 0), 0);
    }

    /// **The ledger sums every live share**: two substrates' stamps make the
    /// total each target is taken against.
    #[test]
    fn the_ledger_sums_every_share() {
        let ledger = WarmLedger::new();
        let zend = ledger.join();
        let npcd = ledger.join();
        zend.hold(GIB);
        assert_eq!(zend.target(4 * GIB), 4 * GIB, "1 GiB alone fits");
        npcd.hold(7 * GIB);
        assert_eq!(
            npcd.target(4 * GIB),
            7 * GIB / 2,
            "8 GiB together: npcd keeps 7/8 of 4 GiB"
        );
        assert_eq!(zend.target(4 * GIB), GIB / 2, "and zend 1/8");
    }

    /// **A substrate that drained stops counting** once it holds its new,
    /// empty figure.
    #[test]
    fn a_drained_substrate_stops_squeezing_the_others() {
        let ledger = WarmLedger::new();
        let zend = ledger.join();
        let npcd = ledger.join();
        zend.hold(3 * GIB);
        npcd.hold(3 * GIB);
        assert_eq!(zend.target(4 * GIB), 2 * GIB);
        npcd.hold(0);
        assert_eq!(zend.target(4 * GIB), 4 * GIB);
    }

    /// **A dropped share stops counting**: a detached substrate's warm bytes
    /// no longer squeeze the ones still attached.
    #[test]
    fn a_dropped_share_leaves_the_total() {
        let ledger = WarmLedger::new();
        let kept = ledger.join();
        let gone = ledger.join();
        gone.hold(8 * GIB);
        drop(gone);
        kept.hold(GIB);
        assert_eq!(kept.target(4 * GIB), 4 * GIB);
        assert_eq!(ledger.shares.lock().unwrap().len(), 2);
        let _again = ledger.join();
        assert_eq!(
            ledger.shares.lock().unwrap().len(),
            2,
            "the next join prunes the dead entry"
        );
    }
}
