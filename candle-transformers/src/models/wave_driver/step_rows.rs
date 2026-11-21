//! The row groups of a co-batched step, and how its forwards are cut at the
//! creep's window.
//!
//! The full-sweep rows cross every layer of the step, while the creep crosses only
//! the layers its window covers. The step is therefore cut into forwards at the
//! window's edges, with the full-sweep rows alone before and after the creep. The
//! creep's residual is held in its own arena between steps. The tier a step places
//! is sized to the widest segment in each layer phase, since the segments share
//! those spans, and to the sum over segments for the forward span, which lives
//! for the whole step.

use super::*;

/// One group of a step's rows: its sequences and their inputs.
#[derive(Clone, Copy)]
pub struct StepRows<'a> {
    pub seqs: &'a [usize],
    pub inputs: &'a [Tensor],
}

impl StepRows<'_> {
    pub const NONE: StepRows<'static> = StepRows {
        seqs: &[],
        inputs: &[],
    };

    /// Caller rows: one per token.
    pub(super) fn tokens(&self) -> usize {
        caller_rows(self.inputs)
    }
}

/// Caller rows of a group's inputs: one per token.
pub(super) fn caller_rows(inputs: &[Tensor]) -> usize {
    inputs
        .iter()
        .map(|t| t.dims().get(1).copied().unwrap_or(1))
        .sum()
}

/// The creep riding a step: its rows, the layers it crosses this step, and the
/// arena its residual is held in between steps.
pub struct StepCreep<'a> {
    pub rows: StepRows<'a>,
    /// First layer this step's window runs.
    pub cursor: usize,
    /// Layer the window stops at.
    pub win_end: usize,
    /// Read when the window resumes (`cursor > 0`); written and committed when
    /// it stops short of the head. A step that needs one and finds none claims
    /// it, sized to the creep's rows, before it places its tier.
    pub residual: &'a mut Option<CreepResidual>,
}

/// One co-batched step: rows that ride every layer, and an optional creep.
pub struct StepSpec<'a> {
    /// One row per sequence, decode kernels.
    pub decode: StepRows<'a>,
    /// Full-sweep rows in the prefill slot: speculative verify blocks.
    pub verify: StepRows<'a>,
    /// Gap-fill rows, which scatter K/V and score nothing.
    pub glue: StepRows<'a>,
    pub creep: Option<StepCreep<'a>>,
}

/// What a step did.
///
/// [`drive_step`] returns `Err` only for a failure of the creep's own segment —
/// its admit or its sweep — after which the creep's window has not run and its
/// arena's committed half is exactly as it was. A failure of a segment that
/// carried only the full-sweep rows lands in [`Self::head`], with
/// [`Self::creep_advanced`] saying whether the creep's window had committed.
pub struct StepOutcome {
    /// The creep's window ran and its residual — or its head — committed.
    pub creep_advanced: bool,
    /// Host wall-clock of the creep's segment, microseconds: the forward the
    /// creep actually rode, apart from the full-sweep rows' segments around it.
    pub creep_us: u64,
    /// The head: logits `[decode | verify | creep]` when the creep reached the
    /// head with the rest, `[decode | verify]` when it stopped short, and none
    /// when nothing rode every layer.
    pub head: Result<WaveResult>,
}

/// One forward of a step: its layer range, and whether the creep rides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    pub start: usize,
    pub end: usize,
    pub creep: bool,
}

/// A step's forwards, in layer order: the full-sweep rows alone up to the
/// creep's window, with the creep across it, alone again after it. Without
/// full-sweep rows, the creep's window alone.
/// The head's rows in the caller's order, as ranges of the head's output: the
/// caller's `n_decode_in` decode rows, then each prefill member's — a folded
/// single-token member (`single`, scored among the decode rows after the
/// caller's) one row, a multi-token one (`multi`, after all of those) its
/// `scored` rows, its last or every row of a verify block — then whatever the
/// head scored past them, the glue rows. `lg_len` is how many the head
/// returned; fewer than the members need is refused.
pub(crate) fn caller_order_rows(
    n_decode_in: usize,
    single: &[usize],
    multi: &[usize],
    scored: &[usize],
    lg_len: usize,
) -> Result<Vec<std::ops::Range<usize>>> {
    let mut member: Vec<std::ops::Range<usize>> = vec![0..0; scored.len()];
    for (r, &j) in single.iter().enumerate() {
        member[j] = n_decode_in + r..n_decode_in + r + 1;
    }
    let mut at = n_decode_in + single.len();
    for &j in multi {
        member[j] = at..at + scored[j];
        at += scored[j];
    }
    if at > lg_len {
        candle::bail!(
            "forward wave: the head scored {lg_len} rows, short of the {at} its decode rows and \
             prefill spans need"
        );
    }
    let mut out = Vec::with_capacity(scored.len() + 2);
    out.push(0..n_decode_in);
    out.extend(member);
    out.push(at..lg_len);
    Ok(out)
}

pub(crate) fn segments(n: usize, cursor: usize, win_end: usize, full: bool) -> Vec<Segment> {
    let seg = |start, end, creep| Segment { start, end, creep };
    if !full {
        return vec![seg(cursor, win_end, true)];
    }
    let mut out = Vec::with_capacity(3);
    if cursor > 0 {
        out.push(seg(0, cursor, false));
    }
    out.push(seg(cursor, win_end, true));
    if win_end < n {
        out.push(seg(win_end, n, false));
    }
    out
}

/// The tier a step places for its segments: each layer phase's span at the
/// widest any segment needs — they reset per layer phase, so the segments
/// share them — and the forward span at the sum, since the step's forward
/// generation does not reset between segments (`docs/wave_step.md` §2.1).
pub(crate) fn step_plan(prices: &[[usize; 3]]) -> [usize; 3] {
    prices.iter().fold([0; 3], |acc, p| {
        [acc[0].max(p[0]), acc[1].max(p[1]), acc[2] + p[2]]
    })
}
