//! Notification when a site goes non-finite.
//!
//! The assert kernel is deliberately asynchronous — it folds statistics into a
//! device slot and nothing reads them until the drain — so by the time a fault
//! is *known*, the wave that produced it has retired and its operands are gone
//! with the arena they lived in. A callback cannot therefore capture the values
//! that caused the fault; what it can do is **arm** something, so the next pass
//! through the offending site captures them while they are still live.
//!
//! That two-phase shape is the whole point:
//!
//! 1. [`report`](super::report) drains, sees site X bad, and invokes the
//!    callbacks with X's [`Finding`].
//! 2. A callback flips a flag.
//! 3. The next execution of X's call site sees the flag and captures its own
//!    inputs and outputs before they are consumed.
//!
//! Faults that reproduce at all reproduce in the dozens, so one is always along
//! after the first.
//!
//! Callbacks run on the draining thread, inside the drain, holding no assert
//! locks. They must not call back into the assert API.

use std::sync::{OnceLock, RwLock};

use super::drain::Finding;

type Callback = Box<dyn Fn(&Finding) + Send + Sync + 'static>;

fn callbacks() -> &'static RwLock<Vec<Callback>> {
    static CBS: OnceLock<RwLock<Vec<Callback>>> = OnceLock::new();
    CBS.get_or_init(|| RwLock::new(Vec::new()))
}

/// Register `f` to run for every site found non-finite by a drain.
///
/// Called once per bad site per drain — and because a drain is followed by an
/// [`epoch`](super::epoch) reset, that is once per bad site per wave rather
/// than once per wave for the whole run.
///
/// Registration is additive and there is no removal: a diagnostic that arms
/// itself has no sensible "off", and the alternative — handing out handles
/// whose drop order matters — buys nothing for a callback that is a flag flip.
pub fn on_bad(f: impl Fn(&Finding) + Send + Sync + 'static) {
    if let Ok(mut v) = callbacks().write() {
        v.push(Box::new(f));
    }
}

/// Whether any callback is registered, so the drain can skip the lock entirely
/// in the overwhelmingly common case of none.
pub fn any_registered() -> bool {
    callbacks().read().map(|v| !v.is_empty()).unwrap_or(false)
}

/// Invoke every registered callback for `finding`. Called by the drain.
pub fn fire(finding: &Finding) {
    let Ok(v) = callbacks().read() else {
        return;
    };
    for f in v.iter() {
        f(finding);
    }
}

#[cfg(test)]
mod tests {
    use super::{any_registered, fire, on_bad};
    use crate::tensor_assert::drain::Finding;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn bad(name: &str) -> Finding {
        Finding {
            name: name.to_string(),
            seq: Some(1),
            nan: 4,
            inf: 0,
            min: Some(-1.0),
            max: Some(1.0),
            elems: 100,
        }
    }

    #[test]
    fn a_registered_callback_sees_the_finding_it_was_fired_with() {
        let hits = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(std::sync::Mutex::new(String::new()));
        let (h, s) = (hits.clone(), seen.clone());
        on_bad(move |f| {
            if f.name == "callback::test::alpha" {
                h.fetch_add(1, Ordering::Relaxed);
                *s.lock().unwrap() = f.name.clone();
            }
        });
        assert!(any_registered());

        fire(&bad("callback::test::alpha"));
        fire(&bad("callback::test::unrelated"));
        fire(&bad("callback::test::alpha"));

        assert_eq!(
            hits.load(Ordering::Relaxed),
            2,
            "fired once per matching finding"
        );
        assert_eq!(*seen.lock().unwrap(), "callback::test::alpha");
    }
}
