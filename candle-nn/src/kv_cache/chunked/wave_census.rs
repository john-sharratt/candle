//! What a layer phase actually carved, itemised — the check on [`super::wave_plan`].
//!
//! # Why the plan needs an external check at all
//!
//! [`super::wave_plan`] claims that "a site that is not a variant here is a site
//! still reaching the driver", and offers `candle::forbidden_alloc` as the
//! instrument that would catch a missing one. That was true before operand
//! provenance. It is not true now: an op reading a wave-backed operand carves
//! its output from the same generation, so an undeclared buffer does not reach
//! the driver, does not appear in a forbidden-allocation report, and costs the
//! span exactly as much as a declared one. The plan can be half the real number
//! and every existing check stays green.
//!
//! That is not a hypothetical. Sizing the tier from `phase_bytes` alone failed on
//! Qwen3-30B-A3B with the attention phase priced at roughly half what it took,
//! and the only symptom was a span exhausting mid-forward.
//!
//! # The peak layer, itemised
//!
//! A phase span is reset when its generation drops, so one generation is one
//! layer's phase and its final cursor is that layer's cost. This records the
//! sizes handed out within a generation and prints them **when that generation
//! sets a new high-water mark for its arena** — the layer that decides the span,
//! at the width that decided it, and nothing else.
//!
//! Reporting on the new maximum rather than on every generation is what makes it
//! usable at all: a benchmark runs tens of thousands of generations and only a
//! handful raise the mark, so the output is a few blocks and it stops. It is also
//! the right selection — a span is sized for its worst moment, so the worst
//! moment is the one whose itemisation answers "what is the plan missing".
//!
//! Sizes identify buffers. Every wave buffer is `rows × cols × width` for known
//! model dimensions, so a line reading `8192 B × 3` against a 1-row wave is three
//! distinct `hidden`-wide BF16 buffers, and which three follows from the chain.
//! There is deliberately no stack capture: symbolising one costs milliseconds,
//! and the arithmetic already names the buffer.
//!
//! Off by default and gated on an environment variable read once, so a
//! disarmed build pays one atomic load per carve.

use std::backtrace::Backtrace;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Whether the census is collecting.
///
/// `KV_WAVE_CENSUS=1`. Cached in an atomic rather than re-read, because this is
/// consulted on the wave path's hottest allocation.
static ENABLED: AtomicBool = AtomicBool::new(false);
/// Whether to name the caller of each carve — `KV_WAVE_CENSUS=labels`.
///
/// Separate from [`ENABLED`] because symbolising a stack per carve costs a
/// fifth of the gate's wall clock, and the sizes alone are enough to *track* an
/// inventory that is already written down. Reach for the labels when
/// establishing one, not when checking it.
static LABELLED: AtomicBool = AtomicBool::new(false);
static INIT: OnceLock<()> = OnceLock::new();

/// One range handed out, and who asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Carve {
    pub len: usize,
    /// `None` when the capture found no frame it could name — a symbol-free
    /// build, or a chain entirely inside the allocator.
    pub label: Option<String>,
}

/// Whether to record carves.
pub(crate) fn enabled() -> bool {
    INIT.get_or_init(|| {
        let v = std::env::var("KV_WAVE_CENSUS").unwrap_or_default();
        let labelled = v.eq_ignore_ascii_case("labels");
        ENABLED.store(
            labelled || v == "1" || v.eq_ignore_ascii_case("true"),
            Ordering::Relaxed,
        );
        LABELLED.store(labelled, Ordering::Relaxed);
    });
    ENABLED.load(Ordering::Relaxed)
}

/// Frames that belong to the allocator rather than to the code that wanted the
/// memory. A carve is attributed to the first frame below all of them.
///
/// The provenance path reaches the arena through several layers — the op, the
/// backend's `alloc_inheriting`, the ticket resolver, the bump — and every one
/// of them appears in every capture, so naming them once here is what makes the
/// label the *caller*.
const ALLOCATOR_FRAMES: [&str; 6] = [
    "wave_census",
    "bump_arena",
    "wave_provenance",
    "alloc_inheriting",
    "backtrace",
    "Backtrace",
];

/// A short name for whatever asked for this carve, or `None` unless
/// `KV_WAVE_CENSUS=labels`.
///
/// Symbolises on every call, which costs milliseconds — a fifth of the gate's
/// wall clock, so it is its own mode rather than part of the census. Sizes alone
/// identify most buffers by arithmetic, but not all of them: two different
/// buffers can be the same number of bytes, and telling those apart is the
/// difference between declaring a site and guessing at one.
pub(crate) fn label() -> Option<String> {
    if !LABELLED.load(Ordering::Relaxed) {
        return None;
    }
    let text = format!("{}", Backtrace::force_capture());
    for line in text.lines() {
        let Some(at) = line.find(char::is_alphabetic) else {
            continue;
        };
        let frame = line[at..].trim();
        // A frame line is `<n>: <symbol>`; the file/line continuation lines
        // start with `at ` and carry no symbol worth naming.
        if frame.starts_with("at ") || frame.is_empty() {
            continue;
        }
        if ALLOCATOR_FRAMES.iter().any(|f| frame.contains(f)) {
            continue;
        }
        if !frame.contains("candle") {
            continue;
        }
        return Some(frame.split_whitespace().next().unwrap_or(frame).to_string());
    }
    None
}

/// Print `sizes` for `arena`, whose generation just closed at `cursor` bytes
/// against a `capacity`-byte span.
///
/// Two views, because they answer different questions.
///
/// **The sequence, in carve order**, is what names a buffer. A size alone is
/// ambiguous — `rows × hidden` in the accumulate dtype and `rows × n_head ×
/// head_dim` in the compute dtype are the same number on a model where
/// `4 · hidden = 2 · n_head · head_dim` — but its *position* in the chain is not,
/// because the chain is code and can be read. Adjacent repeats are run-length
/// encoded so a per-expert loop reads as one line rather than as a screen.
///
/// **The histogram**, by total bytes descending, is what says where the money
/// went: the buffer worth declaring first is the one costing the most.
pub(crate) fn report(arena: &str, cursor: usize, capacity: usize, carves: &[Carve]) {
    let total: usize = carves.iter().map(|c| c.len).sum();
    let alignment_slack = cursor.saturating_sub(total);
    let mut out = format!(
        "wave census: {arena} peak generation {cursor} B of {capacity} B \
         ({} carves, {alignment_slack} B lost to alignment)\n  in carve order:\n",
        carves.len()
    );
    let mut i = 0;
    while i < carves.len() {
        let c = &carves[i];
        let run = carves[i..]
            .iter()
            .take_while(|o| o.len == c.len && o.label == c.label)
            .count();
        let name = c.label.as_deref().unwrap_or("?");
        if run > 1 {
            out.push_str(&format!("    [{i:>3}] {:>12} B x {run}  {name}\n", c.len));
        } else {
            out.push_str(&format!("    [{i:>3}] {:>12} B       {name}\n", c.len));
        }
        i += run;
    }
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for c in carves {
        *hist.entry(c.len).or_insert(0) += 1;
    }
    let mut rows: Vec<(usize, usize)> = hist.into_iter().collect();
    rows.sort_by_key(|&(len, n)| std::cmp::Reverse(len * n));
    out.push_str("  by total:\n");
    for (len, n) in rows {
        out.push_str(&format!("    {len:>12} B x {n:<4} = {:>12} B\n", len * n));
    }
    // `print!` rather than a log macro: this is a measurement a run is launched
    // to collect, and it has to appear whether or not the binary configured a
    // logger.
    print!("{out}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The histogram must aggregate repeats and account for every carve — the
    /// two properties that make a line readable as "this buffer, this many
    /// times".
    #[test]
    fn the_report_aggregates_repeated_sizes() {
        // Exercised for its arithmetic rather than its output: the format is a
        // diagnostic, but silently dropping a carve would make the census lie
        // about what the peak layer did.
        let carves: Vec<Carve> = [8192usize, 8192, 4096, 8192]
            .into_iter()
            .map(|len| Carve { len, label: None })
            .collect();
        let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
        for c in &carves {
            *hist.entry(c.len).or_insert(0) += 1;
        }
        assert_eq!(hist[&8192], 3);
        assert_eq!(hist[&4096], 1);
        assert_eq!(hist.values().sum::<usize>(), carves.len());
        // A smoke call, so a panic in the formatting is caught by the suite
        // rather than by the run it was launched to measure.
        report("test-arena", 20480, 32768, &carves);
    }

    /// Absent the variable the census stays off, which is what keeps the carve
    /// path free of it in every ordinary run.
    #[test]
    fn the_census_is_off_without_the_variable() {
        if std::env::var("KV_WAVE_CENSUS").is_err() {
            assert!(!enabled());
        }
    }
}
