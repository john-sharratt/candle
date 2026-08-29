//! Runtime, in-memory projection-layer kill switch — a DIAGNOSTIC toggle that
//! excludes a layer from projection assembly without touching the schema or
//! persisting anything. Flipping a layer off makes [`super::project::run`] skip
//! it on the next (re)projection; a restart clears every toggle.
//!
//! Distinct from the boot-time `DaemonConfig::disabled_layers`, which only
//! suppresses a layer's startup INGEST — there the layer stays fully projected.
//! This one excludes an *already-populated* layer from the assembled context so
//! you can A/B whether that layer is what's breaking coherence.
//!
//! Keyed by layer NAME — the stable external key, since `LayerId` is only stable
//! within one schema build. Process-global on purpose: the toggle affects the
//! live projection for every conversation, which is exactly the "does turning
//! this layer off restore coherence?" experiment it exists for.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock};

static DISABLED: LazyLock<RwLock<HashSet<String>>> = LazyLock::new(|| RwLock::new(HashSet::new()));
/// Whether [`DISABLED`] is non-empty. Maintained under the write lock, read
/// without any lock — see [`is_layer_disabled`].
static ANY_DISABLED: AtomicBool = AtomicBool::new(false);

/// True while `name` is toggled OFF (excluded from projection assembly).
///
/// Called once per layer per projection assembly, so on the hot path — and the
/// set is empty in every run where nobody has flipped the switch, which is
/// essentially all of them. Taking a process-global `RwLock` and hashing the
/// layer name to discover that cost real CPU: it was the second-hottest resolved
/// symbol on the scheduler's stacks in a full-workspace ingest profile, level
/// with encoding persistence records.
///
/// The atomic short-circuits the empty case. Ordering is `Relaxed` because the
/// lock version was never synchronised against a concurrent toggle either — a
/// flip races the projection already, and lands on the next one.
pub fn is_layer_disabled(name: &str) -> bool {
    if !any_layer_disabled() {
        return false;
    }
    DISABLED.read().map(|s| s.contains(name)).unwrap_or(false)
}

/// Whether ANY layer is toggled off — one relaxed atomic load, no lock.
///
/// Hoistable out of a per-layer loop: a caller walking every layer of a
/// projection tests this once and skips the per-layer calls entirely in the
/// normal case (nothing disabled), instead of re-deriving the same answer for
/// each layer. When it is false, no layer can be disabled, so the per-layer
/// question has a single answer.
pub fn any_layer_disabled() -> bool {
    ANY_DISABLED.load(Ordering::Relaxed)
}

/// Flip `name`'s state; returns the NEW disabled flag (`true` = now excluded).
pub fn toggle_layer(name: &str) -> bool {
    let mut set = DISABLED.write().unwrap_or_else(|e| e.into_inner());
    let now_disabled = if set.remove(name) {
        false
    } else {
        set.insert(name.to_string());
        true
    };
    // Published under the write lock, so the flag can never claim "nothing
    // disabled" while the set holds an entry.
    ANY_DISABLED.store(!set.is_empty(), Ordering::Relaxed);
    now_disabled
}

/// Snapshot of every currently-disabled layer name — for surfacing UI state.
pub fn disabled_layers() -> HashSet<String> {
    DISABLED.read().map(|s| s.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The toggle is process-global, so tests that touch it cannot run
    /// concurrently: one test's entry makes another's "is the set empty?"
    /// assertion false. Serialise them rather than weakening the assertions —
    /// the empty-set state is exactly what the fast path keys on.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn toggle_flips_and_reports() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Use a name unlikely to collide with any real layer under test.
        let n = "__layer_toggle_unit_test__";
        assert!(!is_layer_disabled(n));
        assert!(toggle_layer(n), "first toggle disables");
        assert!(is_layer_disabled(n));
        assert!(disabled_layers().contains(n));
        assert!(!toggle_layer(n), "second toggle re-enables");
        assert!(!is_layer_disabled(n));
    }

    /// The empty-set fast path is a process-global flag, so it must track the
    /// set's emptiness and not any one name: while ANY layer is disabled the
    /// flag is on and every query has to consult the set, and the flag only
    /// clears once the LAST entry goes.
    #[test]
    fn fast_path_flag_tracks_the_set_not_one_name() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let a = "__toggle_fast_path_a__";
        let b = "__toggle_fast_path_b__";
        assert!(toggle_layer(a));
        assert!(toggle_layer(b));
        // Both held: neither may be reported enabled by a stale global flag.
        assert!(is_layer_disabled(a));
        assert!(is_layer_disabled(b));
        // One released — the other must still read as disabled, which only
        // works if the flag stayed on for the remaining entry.
        assert!(!toggle_layer(a));
        assert!(!is_layer_disabled(a));
        assert!(is_layer_disabled(b), "flag cleared while an entry remained");
        // Last one released — the fast path may now short-circuit.
        assert!(!toggle_layer(b));
        assert!(!is_layer_disabled(b));
        assert!(disabled_layers().is_empty());
    }
}
