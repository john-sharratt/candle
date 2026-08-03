//! Process-global ring of recent **VRAM lifecycle actions** — defrag moves,
//! evictions, pool trims, empty-arena releases, cold→hot elevations, hot→warm
//! migrations, segment-maintenance applies — dumped alongside the per-thread
//! kernel-launch breadcrumbs when a CUDA fault poisons the context.
//!
//! The kernel ring answers "which kernel faulted"; this ring answers "what was
//! MOVING memory around it, on which thread, how recently". Illegal-address
//! faults in this codebase have historically come from a lifecycle action
//! (trim, compaction, eviction) racing in-flight kernel reads across threads —
//! attribution needs the interleaving, and WDDM makes launch-blocking
//! attribution unreliable, so the interleaving must come from the crash report
//! itself.
//!
//! Global (unlike the thread-local kernel ring) because the mover and the
//! faulting kernel are usually on different threads: relief runs on the
//! scheduler thread, migrations on the persistence thread. A `Mutex` is fine —
//! entries are recorded at most a handful of times per wave, never per token.

use std::sync::Mutex;
use std::time::Instant;

const RING_LEN: usize = 48;

#[derive(Clone, Copy)]
struct Entry {
    at: Option<Instant>,
    /// Which thread lane recorded it (`"sched"` / `"tier"` / …).
    lane: &'static str,
    op: &'static str,
    a: u64,
    b: u64,
}

const EMPTY: Entry = Entry {
    at: None,
    lane: "",
    op: "",
    a: 0,
    b: 0,
};

static RING: Mutex<([Entry; RING_LEN], usize)> = Mutex::new(([EMPTY; RING_LEN], 0));

/// Record one lifecycle action. `a` / `b` are op-specific magnitudes
/// (bytes, counts, before/after) — see each call site's op string.
pub fn note(lane: &'static str, op: &'static str, a: u64, b: u64) {
    let mut g = RING.lock().unwrap_or_else(|e| e.into_inner());
    let (ring, pos) = &mut *g;
    ring[*pos] = Entry {
        at: Some(Instant::now()),
        lane,
        op,
        a,
        b,
    };
    *pos = (*pos + 1) % RING_LEN;
}

/// Dump the ring newest-first with per-entry age, for the poison report.
pub fn dump() -> String {
    let g = RING.lock().unwrap_or_else(|e| e.into_inner());
    let (ring, pos) = &*g;
    let now = Instant::now();
    let mut out = Vec::new();
    for k in 0..RING_LEN {
        let i = (pos + RING_LEN - 1 - k) % RING_LEN;
        let e = ring[i];
        let Some(at) = e.at else { continue };
        out.push(format!(
            "    #{k} [{:>7}ms ago] {:<5} {} a={} b={}",
            now.duration_since(at).as_millis(),
            e.lane,
            e.op,
            e.a,
            e.b,
        ));
    }
    if out.is_empty() {
        "(no VRAM lifecycle actions recorded)".to_string()
    } else {
        format!(
            "recent VRAM lifecycle actions (all threads), newest first — an \
             illegal-address fault is usually a kernel racing one of these:\n{}",
            out.join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{dump, note};

    #[test]
    fn ring_records_and_dumps_newest_first() {
        note("sched", "test_op_a", 1, 2);
        note("tier", "test_op_b", 3, 4);
        let d = dump();
        let a = d.find("test_op_a").expect("op_a present");
        let b = d.find("test_op_b").expect("op_b present");
        assert!(b < a, "newest (op_b) must be listed before op_a");
        assert!(d.contains("a=3 b=4"));
    }
}
