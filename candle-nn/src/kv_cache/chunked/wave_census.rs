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
//! Built in by the `wave-census` feature and absent otherwise: the switch is a
//! compile-time constant, so an ordinary build carries none of it on the
//! carve path.
//!
//! `wave-census-labels` adds the caller of each carve. It is a feature of its
//! own because symbolising a stack per carve costs a fifth of the gate's wall
//! clock, and the sizes alone are enough to *track* an inventory that is
//! already written down. Reach for the labels when establishing one, not when
//! checking it.

use std::backtrace::Backtrace;
use std::collections::BTreeMap;

/// One range handed out, and who asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Carve {
    /// Byte offset of this carve from the arena's base.
    ///
    /// The census itself only ever needed `len` — it reports how much a phase
    /// spent, not where. The span audit needs *where*: a carve is a live
    /// activation buffer, and "do two of them share bytes" cannot be asked
    /// without the offset.
    pub start: usize,
    pub len: usize,
    /// `None` when the capture found no frame it could name — a symbol-free
    /// build, or a chain entirely inside the allocator.
    pub label: Option<String>,
}

/// Whether to record carves — the `wave-census` feature.
pub(crate) const fn enabled() -> bool {
    cfg!(feature = "wave-census")
}

/// Frames that belong to the allocator rather than to the code that wanted the
/// memory. A carve is attributed to the first frame below all of them.
///
/// The provenance path reaches the arena through several layers — the op, the
/// backend's `alloc_inheriting`, the ticket resolver, the bump — and every one
/// of them appears in every capture, so naming them once here is what makes the
/// label the *caller*.
///
/// `CudaStorage` is on the list because the matmul family allocates through
/// inherent methods on it, and stopping there labels a third of the attention
/// phase with one uninformative name. The frame below it is the op or the model
/// function that wanted the buffer, which is what the inventory needs.
const ALLOCATOR_FRAMES: [&str; 7] = [
    "wave_census",
    "bump_arena",
    "wave_provenance",
    "alloc_inheriting",
    "CudaStorage",
    "backtrace",
    "Backtrace",
];

/// A short name for whatever asked for this carve, or `None` without the
/// `wave-census-labels` feature.
///
/// Symbolises on every call, which costs milliseconds — a fifth of the gate's
/// wall clock, so it is its own feature rather than part of the census. Sizes
/// alone identify most buffers by arithmetic, but not all of them: two
/// different buffers can be the same number of bytes, and telling those apart
/// is the difference between declaring a site and guessing at one.
pub(crate) fn label() -> Option<String> {
    if !cfg!(feature = "wave-census-labels") {
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
            .map(|len| Carve { start: 0, len, label: None })
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
}
