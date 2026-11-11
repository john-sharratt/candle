//! The engine's high-rate events, counted rather than logged.
//!
//! A reprojection's splice plan, a bulk prefill launch, a sequence's
//! construction, a stencil finishing: each happens many times a second under
//! load, and written as a log line each one is what fills the size-bounded log
//! in minutes and rotates every other line out of it. Here each kind keeps its
//! count since start and a recent instance, as text, with when it happened —
//! what anyone reading the line actually wanted from it: how often, and what
//! one looked like. `GET /v1/events` serves [`snapshot`].
//!
//! **Cheap where it fires most.** A kind's instance is formatted at most once
//! every [`SAMPLE_EVERY`]; between, a note is a count under the table's lock
//! and no `format!` at all, so a note on every layer of every forward costs a
//! counter, not a string.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// How often a kind's instance is taken: more often only counts.
pub const SAMPLE_EVERY: Duration = Duration::from_millis(100);

/// One kind's record.
struct Entry {
    count: u64,
    last: String,
    at: Instant,
}

static EVENTS: OnceLock<Mutex<HashMap<&'static str, Entry>>> = OnceLock::new();

fn events() -> MutexGuard<'static, HashMap<&'static str, Entry>> {
    EVENTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Count one `kind` event, and take `detail` as its instance when the last
/// one taken is [`SAMPLE_EVERY`] old — formatted outside the table's lock.
pub fn note(kind: &'static str, detail: impl FnOnce() -> String) {
    let due = {
        let mut table = events();
        match table.get_mut(kind) {
            Some(entry) => {
                entry.count += 1;
                entry.at.elapsed() >= SAMPLE_EVERY
            }
            None => {
                table.insert(
                    kind,
                    Entry {
                        count: 1,
                        last: String::new(),
                        at: Instant::now(),
                    },
                );
                true
            }
        }
    };
    if !due {
        return;
    }
    let last = detail();
    if let Some(entry) = events().get_mut(kind) {
        entry.last = last;
        entry.at = Instant::now();
    }
}

/// One kind as [`snapshot`] reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct EventCount {
    pub kind: &'static str,
    /// Events of this kind since the process started.
    pub count: u64,
    /// A recent one, taken at most [`SAMPLE_EVERY`] apart.
    pub last: String,
    /// Seconds since that one was taken.
    pub age_s: f64,
}

/// Every kind noted so far, most frequent first.
pub fn snapshot() -> Vec<EventCount> {
    let table = events();
    let mut out: Vec<EventCount> = table
        .iter()
        .map(|(kind, e)| EventCount {
            kind,
            count: e.count,
            last: e.last.clone(),
            age_s: e.at.elapsed().as_secs_f64(),
        })
        .collect();
    out.sort_by(|a, b| b.count.cmp(&a.count).then(a.kind.cmp(b.kind)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(kind: &str) -> EventCount {
        snapshot().into_iter().find(|e| e.kind == kind).unwrap()
    }

    /// **A kind counts every event and takes an instance at most every
    /// [`SAMPLE_EVERY`]**: two notes in a row count two and keep the first's
    /// text; one past the interval is taken. Kinds named only by this test, so
    /// the process-wide table other tests write to does not disturb it.
    #[test]
    fn a_kind_counts_every_event_and_samples_its_instance() {
        note("test:engine_events:a", || "first".into());
        note("test:engine_events:a", || "second".into());
        let a = of("test:engine_events:a");
        assert_eq!((a.count, a.last.as_str()), (2, "first"));
        std::thread::sleep(SAMPLE_EVERY + Duration::from_millis(20));
        note("test:engine_events:a", || "third".into());
        let a = of("test:engine_events:a");
        assert_eq!((a.count, a.last.as_str()), (3, "third"));
    }

    /// The more frequent kind is listed first.
    #[test]
    fn the_more_frequent_kind_comes_first() {
        for _ in 0..3 {
            note("test:engine_events:often", String::new);
        }
        note("test:engine_events:once", String::new);
        let snap = snapshot();
        let at = |kind: &str| snap.iter().position(|e| e.kind == kind).unwrap();
        assert!(at("test:engine_events:often") < at("test:engine_events:once"));
    }
}
