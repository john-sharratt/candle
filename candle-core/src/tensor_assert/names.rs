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
    /// The call site's own `&'static str`, kept rather than copied.
    ///
    /// A consumer that arms a capture compares the armed site against the call
    /// site **by pointer**, so the table has to hand back the very string the
    /// call site passes — a leaked copy of equal content would compare unequal
    /// and the arm could never fire. See [`interned`].
    by_name: HashMap<&'static str, usize>,
    by_slot: Vec<&'static str>,
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

/// Indexed site names, leaked so a call site can hold a `&'static str`. See
/// [`site`]. Recovering a name from a finding goes through [`interned`], which
/// reads the slot table instead — this one holds only the indexed names.
type SiteTable = RwLock<HashMap<(&'static str, usize), &'static str>>;
static SITES: OnceLock<SiteTable> = OnceLock::new();

/// The slot index for `name`, assigning one on first sight.
///
/// Returns `None` once [`MAX_SLOTS`] distinct names have been seen. Silently
/// folding a further name into an existing slot would mix two tensors'
/// statistics into one report, which is worse than not measuring it — so this
/// says so, once, and then declines.
/// `name` is `&'static str` because the table retains it — see [`Names::by_name`]
/// and [`interned`]. Every call site already passes a literal or a [`site`]
/// result, so this costs no caller anything and refuses, at compile time, the
/// `&format!(...)` that the harness forbids anyway.
pub fn slot_for(name: &'static str) -> Option<usize> {
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
    n.by_name.insert(name, idx);
    n.by_slot.push(name);
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

/// The call site's `&'static str` for any name that has been asserted.
///
/// A [`Finding`](super::Finding) carries an owned `String`, so a consumer that
/// wants to compare against a call site by POINTER — the only comparison cheap
/// enough to sit on a hot path — needs the original back.
///
/// This reads the **slot** table, which holds every name ever asserted, rather
/// than the [`site`] table, which holds only the indexed ones. That distinction
/// was a live bug, and an expensive one: the capture in `nan_capture` drops any
/// finding this returns `None` for, so while it consulted `SITES` the arm could
/// land *only* on a `site()`-built name. A run whose cascade began at the plain
/// literal `attn.ctx_raw` (`seq=1`) skipped it and every other literal, arming
/// `moe.shared_gated.L31` at `seq=15` — the fifteenth consequence — and three
/// successive captures faithfully dumped the wrong tensor's operands.
///
/// Returns `None` only for a name that has never been asserted at all.
pub fn interned(name: &str) -> Option<&'static str> {
    let n = names().read().ok()?;
    n.by_name.get_key_value(name).map(|(k, _)| *k)
}

/// The name a slot was registered under, for the drain's report.
pub fn name_of(idx: usize) -> Option<String> {
    let n = names().read().ok()?;
    n.by_slot.get(idx).map(|s| s.to_string())
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
    use super::{claim_once, interned, name_of, site, slot_for};

    #[test]
    fn a_name_keeps_its_slot_and_distinct_names_get_distinct_slots() {
        let a = slot_for("tensor_assert::test::alpha").expect("slot");
        let b = slot_for("tensor_assert::test::beta").expect("slot");
        assert_ne!(a, b);
        assert_eq!(slot_for("tensor_assert::test::alpha"), Some(a));
        assert_eq!(name_of(a).as_deref(), Some("tensor_assert::test::alpha"));
    }

    /// A capture's arm is dropped for any finding [`interned`] cannot resolve,
    /// so a name it fails on can never be captured — only ranked. While this
    /// consulted the `site()` table alone, that silently excluded every plain
    /// literal, which is most of the instrumented forward: the arm slid past the
    /// origin to the first indexed name downstream of it.
    #[test]
    fn a_plain_literal_is_recoverable_by_pointer_not_only_an_indexed_site() {
        const LITERAL: &str = "tensor_assert::test::plain_literal";
        slot_for(LITERAL).expect("slot");
        let back = interned(LITERAL).expect("a literal that has been asserted must resolve");
        assert!(
            std::ptr::eq(back.as_ptr(), LITERAL.as_ptr()),
            "must hand back the call site's own string, so the arm compares equal by pointer"
        );

        // The indexed form keeps working, and resolves to `site`'s own leak.
        let indexed = site("tensor_assert::test::indexed.L", 31);
        slot_for(indexed).expect("slot");
        let back = interned("tensor_assert::test::indexed.L31").expect("indexed must resolve");
        assert!(std::ptr::eq(back.as_ptr(), indexed.as_ptr()));

        assert_eq!(interned("tensor_assert::test::never_asserted"), None);
    }

    #[test]
    fn the_once_latch_fires_exactly_once() {
        let s = slot_for("tensor_assert::test::once").expect("slot");
        assert!(claim_once(s), "first claim must fire");
        assert!(!claim_once(s), "second claim must not fire");
        assert!(!claim_once(s), "third claim must not fire");
    }
}
