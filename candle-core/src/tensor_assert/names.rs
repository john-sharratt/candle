//! Host-side name → slot-index interning.
//!
//! An assert is written as `x.assert("moe.router_logits")`, so a name has to
//! become a slot index on every call. That lookup is host-only — a read lock and
//! a hash — and issues no device work, which is why it is allowed to sit on the
//! hot path at all.
//!
//! Slots are never recycled. A name that is asserted in one wave and not the
//! next keeps its slot and simply stops accumulating, so a drained report can
//! always name every slot it prints.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

use super::slots::MAX_SLOTS;

struct Names {
    by_name: HashMap<String, usize>,
    by_slot: Vec<String>,
    /// Per-slot latch for [`super::should_run_once`]; a weight is checked the
    /// first time and never again.
    once: Vec<bool>,
}

fn names() -> &'static RwLock<Names> {
    static N: OnceLock<RwLock<Names>> = OnceLock::new();
    N.get_or_init(|| {
        RwLock::new(Names {
            by_name: HashMap::new(),
            by_slot: Vec::new(),
            once: Vec::new(),
        })
    })
}

/// Whether the table has already overflowed, so the warning is emitted once
/// rather than once per assert after the table fills.
static OVERFLOWED: AtomicBool = AtomicBool::new(false);

/// Indexed site names, leaked so a call site can hold a `&'static str` and a
/// consumer can compare against it by pointer. See [`site`] and
/// [`interned_site`].
type SiteTable = RwLock<HashMap<(&'static str, usize), &'static str>>;
static SITES: OnceLock<SiteTable> = OnceLock::new();

/// The slot index for `name`, assigning one on first sight.
///
/// Returns `None` once [`MAX_SLOTS`] distinct names have been seen. Silently
/// folding a further name into an existing slot would mix two tensors'
/// statistics into one report, which is worse than not measuring it — so this
/// says so, once, and then declines.
pub fn slot_for(name: &str) -> Option<usize> {
    {
        let n = names().read().ok()?;
        if let Some(&idx) = n.by_name.get(name) {
            return Some(idx);
        }
    }
    let mut n = names().write().ok()?;
    // Re-check: another thread may have inserted between the two locks.
    if let Some(&idx) = n.by_name.get(name) {
        return Some(idx);
    }
    let idx = n.by_slot.len();
    if idx >= MAX_SLOTS {
        if !OVERFLOWED.swap(true, Ordering::Relaxed) {
            tracing::error!(
                target: "candle_core::tensor_assert",
                max_slots = MAX_SLOTS,
                first_dropped = %name,
                "tensor_assert: out of slots — further assert names are not measured"
            );
        }
        return None;
    }
    n.by_name.insert(name.to_string(), idx);
    n.by_slot.push(name.to_string());
    n.once.push(false);
    Some(idx)
}

/// A stable `&'static str` for an indexed site — `site("qwen35.layer_out.L", 7)`
/// is `"qwen35.layer_out.L7"`.
///
/// Per-layer asserts need a name per layer, and formatting one per call per
/// layer per wave would put host allocation on the hot path. The names are
/// built once, on first sight, and leaked: they are bounded by [`MAX_SLOTS`],
/// they live as long as the report that prints them, and leaking is what lets
/// the caller pass a `&'static str` without keeping a lock held across the
/// assert.
pub fn site(prefix: &'static str, idx: usize) -> &'static str {
    let table = SITES.get_or_init(|| RwLock::new(HashMap::new()));
    if let Ok(t) = table.read() {
        if let Some(&s) = t.get(&(prefix, idx)) {
            return s;
        }
    }
    let owned: &'static str = Box::leak(format!("{prefix}{idx}").into_boxed_str());
    match table.write() {
        // Another thread may have raced us here; keep whichever landed first so
        // one site is one string, and let the loser's allocation go.
        Ok(mut t) => t.entry((prefix, idx)).or_insert(owned),
        Err(_) => owned,
    }
}

/// The interned `&'static str` for a name previously produced by [`site`].
///
/// A [`Finding`](super::Finding) carries an owned `String`, so a consumer that
/// wants to compare against a call site by POINTER — the only comparison cheap
/// enough to sit on a hot path — needs the original back. Returns `None` for a
/// name that never came from `site`, which is every plain string literal.
pub fn interned_site(name: &str) -> Option<&'static str> {
    let table = SITES.get()?;
    let t = table.read().ok()?;
    t.values().find(|s| **s == name).copied()
}

/// The name a slot was registered under, for the drain's report.
pub fn name_of(idx: usize) -> Option<String> {
    let n = names().read().ok()?;
    n.by_slot.get(idx).cloned()
}

/// How many slots have been claimed, so the drain reads no further.
pub fn claimed() -> usize {
    names().read().map(|n| n.by_slot.len()).unwrap_or(0)
}

/// Claim `idx`'s one-shot latch: `true` the first time, `false` after.
///
/// This is what makes `assert_once` on a weight affordable. A weight does not
/// change between forwards, so re-reading it every layer of every wave would be
/// exactly the bandwidth perturbation the whole design exists to avoid.
pub fn claim_once(idx: usize) -> bool {
    let mut n = match names().write() {
        Ok(n) => n,
        Err(_) => return false,
    };
    match n.once.get_mut(idx) {
        Some(fired) if !*fired => {
            *fired = true;
            true
        }
        _ => false,
    }
}

/// Re-arm every one-shot latch, so a new epoch re-checks the weights.
pub fn rearm_once() {
    if let Ok(mut n) = names().write() {
        n.once.iter_mut().for_each(|f| *f = false);
    }
}

#[cfg(test)]
mod tests {
    use super::{claim_once, name_of, slot_for};

    #[test]
    fn a_name_keeps_its_slot_and_distinct_names_get_distinct_slots() {
        let a = slot_for("tensor_assert::test::alpha").expect("slot");
        let b = slot_for("tensor_assert::test::beta").expect("slot");
        assert_ne!(a, b);
        assert_eq!(slot_for("tensor_assert::test::alpha"), Some(a));
        assert_eq!(name_of(a).as_deref(), Some("tensor_assert::test::alpha"));
    }

    #[test]
    fn the_once_latch_fires_exactly_once() {
        let s = slot_for("tensor_assert::test::once").expect("slot");
        assert!(claim_once(s), "first claim must fire");
        assert!(!claim_once(s), "second claim must not fire");
        assert!(!claim_once(s), "third claim must not fire");
    }
}
