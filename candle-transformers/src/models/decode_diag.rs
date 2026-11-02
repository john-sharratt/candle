//! What one armed decode attended, for the capped diagnostic decode.
//!
//! A sparse-attention model reads a few thousand cells of a context of hundreds
//! of thousands, so a piece can be placed, indexed and correct and still never
//! be read. Nothing outside the decode can tell that apart from a piece that
//! was read and ignored. Armed, this records for each watched query what the
//! selection chose: per piece of the slot, how many cells it took, how its
//! blocks ranked, and the score a block had to beat.
//!
//! **It only observes.** It reads the kernel's own scores and table after they
//! are written, and changes nothing a decode computes, unless the caller asks
//! for a [`Control`]. Armed, it fences on every read and holds the card to one
//! timeline, which is why only a decode run on purpose under a cap is ever
//! armed.
//!
//! **Outside a diagnostic call a hook costs one not-taken branch on a relaxed
//! load: no lock, no allocation, no launch, no readback.** Every hook on the
//! decode path is written `if decode_diag::armed() { .. }`, or begins with
//! that test, and everything past it is `#[cold]`. Nothing may be computed for
//! a hook ahead of the test.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

/// Blocks a record lists by name, best first.
const TOP_BLOCKS: usize = 24;

/// The per-piece rows one record keeps, most attended first. A slot holds
/// hundreds of pieces and the rest are counted in [`LayerRecord::pieces_unread`].
const PIECES_KEPT: usize = 64;

/// One piece of the slot: positions `start..end`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PieceSpan {
    pub label: String,
    pub start: usize,
    pub end: usize,
}

/// What a selection did with one piece.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PieceAttention {
    pub label: String,
    /// Candidate blocks inside the piece.
    pub blocks: usize,
    pub attended_blocks: usize,
    pub attended_cells: usize,
    pub best_score: f32,
    /// The 0-based rank of the piece's best block among every candidate.
    pub best_rank: usize,
}

/// One block a record names.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TopBlock {
    pub position: usize,
    pub score: f32,
    pub attended: bool,
    pub piece: Option<String>,
}

/// One query's selection at one attention layer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerRecord {
    /// The decode step, as the scheduler counted it.
    pub step: u64,
    /// The layer's index among the attention layers.
    pub kv: usize,
    pub query_position: usize,
    pub candidates: usize,
    pub attended_blocks: usize,
    pub attended_cells: usize,
    /// The lowest score of an attended block ranked on score: what a block had
    /// to beat. Blocks at zero score are not counted, as a recent span's are.
    pub cutoff: f32,
    pub pieces: Vec<PieceAttention>,
    /// Pieces that took no cell and are not listed.
    pub pieces_unread: usize,
    pub top: Vec<TopBlock>,
}

/// What a control changes about the armed decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum Control {
    /// The decode as production runs it.
    #[default]
    Sparse,
    /// Attend every cell: the wave builds no selection.
    Dense,
    /// Place every piece anew at each reprojection, splicing nothing.
    Fresh,
}

/// One decoded token.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepRecord {
    pub step: u64,
    /// Rows in the wave that produced it.
    pub rows: usize,
    /// What the sampler chose, before any recording replaced it.
    pub sampled: u32,
    /// The best ids and their logits, best first.
    pub top: Vec<(u32, f32)>,
    /// A digest of the whole row: two steps are the same decode when it agrees.
    pub logits_hash: u64,
}

/// One reprojection drained during the armed decode.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReprojectRecord {
    pub step: u64,
    pub ms: f64,
}

/// Everything one armed decode recorded.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Capture {
    pub layers: Vec<LayerRecord>,
    pub steps: Vec<StepRecord>,
    pub reprojects: Vec<ReprojectRecord>,
    /// The slot's pieces as the decode finished on them.
    pub pieces: Vec<PieceSpan>,
}

/// What to hold and watch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arm {
    pub control: Control,
    /// The one timeline that decodes while armed.
    pub timeline: u64,
    /// The generated-token count recording starts from.
    pub from: usize,
    /// The generated-token count the closing live reprojection falls on; the
    /// slot it builds is not the one the decode read.
    pub live_at: usize,
}

struct Watch {
    arm: Arm,
    seqs: HashSet<usize>,
    step: u64,
    rows: usize,
    capture: Capture,
}

static ARMED: AtomicBool = AtomicBool::new(false);
static WATCH: Mutex<Option<Watch>> = Mutex::new(None);

fn watch<R>(f: impl FnOnce(&mut Option<Watch>) -> R) -> R {
    f(&mut WATCH.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Whether a diagnostic decode is armed. The one cost an unarmed decode pays.
#[inline]
pub fn armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Arm the capture. Refused while another holds it: two diagnostic decodes
/// would read each other's records.
pub fn arm(arm: Arm) -> bool {
    watch(|w| {
        if w.is_some() {
            return false;
        }
        *w = Some(Watch {
            arm,
            seqs: HashSet::new(),
            step: 0,
            rows: 0,
            capture: Capture::default(),
        });
        ARMED.store(true, Ordering::Relaxed);
        true
    })
}

/// End the capture and hand over what it recorded.
pub fn disarm() -> Capture {
    ARMED.store(false, Ordering::Relaxed);
    watch(|w| w.take().map(|w| w.capture).unwrap_or_default())
}

/// What the capture was armed with.
#[cold]
pub fn armed_with() -> Option<Arm> {
    watch(|w| w.as_ref().map(|w| w.arm))
}

/// This step: the sequences it records, its generated count, its wave's rows.
#[cold]
pub fn set_step(seqs: impl IntoIterator<Item = usize>, step: u64, rows: usize) {
    watch(|w| {
        if let Some(w) = w {
            w.seqs = seqs.into_iter().collect();
            w.step = step;
            w.rows = rows;
        }
    });
}

/// The slot's pieces as they stand, for naming what a selection chose.
#[cold]
pub fn set_pieces(pieces: Vec<PieceSpan>) {
    watch(|w| {
        if let Some(w) = w {
            w.capture.pieces = pieces;
        }
    });
}

/// Whether `seq`'s selections are being watched.
#[cold]
pub fn watching(seq: usize) -> bool {
    watch(|w| w.as_ref().is_some_and(|w| w.seqs.contains(&seq)))
}

#[cold]
fn control_is(control: Control) -> bool {
    watch(|w| w.as_ref().is_some_and(|w| w.arm.control == control))
}

/// Whether the wave should attend every cell.
#[inline]
pub fn dense() -> bool {
    armed() && control_is(Control::Dense)
}

/// Whether a reprojection should place every piece anew.
#[inline]
pub fn fresh() -> bool {
    armed() && control_is(Control::Fresh)
}

/// Keep `record`.
fn keep(record: LayerRecord) {
    watch(|w| {
        if let Some(w) = w {
            w.capture.layers.push(record);
        }
    });
}

/// Times one drain of reprojections while it lives.
pub struct ReprojectSpan(Instant);

impl Drop for ReprojectSpan {
    fn drop(&mut self) {
        let ms = self.0.elapsed().as_secs_f64() * 1000.0;
        watch(|w| {
            if let Some(w) = w {
                let step = w.step;
                w.capture.reprojects.push(ReprojectRecord { step, ms });
            }
        });
    }
}

/// A timer for a drain of reprojections, when armed.
#[inline]
pub fn reproject_span() -> Option<ReprojectSpan> {
    armed().then(|| ReprojectSpan(Instant::now()))
}

/// How many ids a [`StepRecord`] names.
const TOP_IDS: usize = 8;

/// FNV-1a over the bits of `logits`.
fn digest(logits: &[f32]) -> u64 {
    logits.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Reduce one row of logits to a [`StepRecord`].
fn reduce_step(logits: &[f32], sampled: u32, step: u64, rows: usize) -> StepRecord {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    StepRecord {
        step,
        rows,
        sampled,
        top: order
            .iter()
            .take(TOP_IDS)
            .map(|&i| (i as u32, logits[i]))
            .collect(),
        logits_hash: digest(logits),
    }
}

/// Record the row that produced `sampled` for sequence `seq`.
#[cold]
pub fn record_step(seq: usize, logits: &[f32], sampled: u32) {
    watch(|w| {
        if let Some(w) = w {
            if w.seqs.contains(&seq) {
                let record = reduce_step(logits, sampled, w.step, w.rows);
                w.capture.steps.push(record);
            }
        }
    });
}

/// The piece holding `position`.
fn piece_of(pieces: &[PieceSpan], position: usize) -> Option<usize> {
    let i = pieces.partition_point(|p| p.end <= position);
    pieces.get(i).filter(|p| p.start <= position).map(|_| i)
}

/// One query's raw selection, as the kernel left it.
pub struct RawSelection<'a> {
    pub kv: usize,
    pub query_position: usize,
    /// Each candidate block's score, indexed by block.
    pub scores: &'a [f32],
    /// The row's selection, as `(block, cells)`.
    pub entries: &'a [(usize, usize)],
    /// The first position of block `b`.
    pub block_start: &'a dyn Fn(usize) -> usize,
}

/// Reduce `raw` to one record over `pieces`, in position order.
pub fn reduce(raw: &RawSelection<'_>, pieces: &[PieceSpan], step: u64) -> LayerRecord {
    let n = raw.scores.len();
    let mut cells = vec![0usize; n];
    for &(block, width) in raw.entries {
        if let Some(c) = cells.get_mut(block) {
            *c = (*c).max(width);
        }
    }
    // Rank by score, ties to the earlier block, as the kernel's key does.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| raw.scores[b].total_cmp(&raw.scores[a]).then(a.cmp(&b)));
    let mut rank = vec![0usize; n];
    for (r, &b) in order.iter().enumerate() {
        rank[b] = r;
    }
    let mut rows: Vec<PieceAttention> = pieces
        .iter()
        .map(|p| PieceAttention {
            label: p.label.clone(),
            blocks: 0,
            attended_blocks: 0,
            attended_cells: 0,
            best_score: f32::NEG_INFINITY,
            best_rank: usize::MAX,
        })
        .collect();
    for b in 0..n {
        let Some(i) = piece_of(pieces, (raw.block_start)(b)) else {
            continue;
        };
        let row = &mut rows[i];
        row.blocks += 1;
        row.best_score = row.best_score.max(raw.scores[b]);
        row.best_rank = row.best_rank.min(rank[b]);
        if cells[b] > 0 {
            row.attended_blocks += 1;
            row.attended_cells += cells[b];
        }
    }
    let cutoff = (0..n)
        .filter(|&b| cells[b] > 0 && raw.scores[b] > 0.0)
        .map(|b| raw.scores[b])
        .fold(f32::INFINITY, f32::min);
    let top = order
        .iter()
        .take(TOP_BLOCKS)
        .map(|&b| {
            let position = (raw.block_start)(b);
            TopBlock {
                position,
                score: raw.scores[b],
                attended: cells[b] > 0,
                piece: piece_of(pieces, position).map(|i| pieces[i].label.clone()),
            }
        })
        .collect();
    rows.retain(|r| r.blocks > 0);
    rows.sort_by(|a, b| {
        b.attended_cells
            .cmp(&a.attended_cells)
            .then(a.best_rank.cmp(&b.best_rank))
    });
    let unread = rows.iter().filter(|r| r.attended_cells == 0).count();
    // Every piece that took a cell, then the nearest misses, up to the cap.
    let kept = rows.len().min(PIECES_KEPT).max(rows.len() - unread);
    let listed_unread = kept - (rows.len() - unread);
    rows.truncate(kept);
    LayerRecord {
        step,
        kv: raw.kv,
        query_position: raw.query_position,
        candidates: n,
        attended_blocks: cells.iter().filter(|&&c| c > 0).count(),
        attended_cells: cells.iter().sum(),
        cutoff: if cutoff.is_finite() { cutoff } else { 0.0 },
        pieces: rows,
        pieces_unread: unread - listed_unread,
        top,
    }
}

/// Record `raw` for the armed run.
#[cold]
pub fn record(raw: &RawSelection<'_>) {
    let (pieces, step) = watch(|w| {
        w.as_ref()
            .map_or((Vec::new(), 0), |w| (w.capture.pieces.clone(), w.step))
    });
    keep(reduce(raw, &pieces, step));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces() -> Vec<PieceSpan> {
        vec![
            PieceSpan {
                label: "a".into(),
                start: 0,
                end: 8,
            },
            PieceSpan {
                label: "b".into(),
                start: 8,
                end: 16,
            },
            PieceSpan {
                label: "c".into(),
                start: 16,
                end: 24,
            },
        ]
    }

    /// **A record says which piece took the cells, what it had to beat, and
    /// how near the unread piece came**, on values worked by hand. Blocks are 4
    /// wide; `b` holds the two best blocks and is attended, `c`'s best is third
    /// and is not, and `a` is attended on a score of 0.25.
    #[test]
    fn a_selection_reduces_to_its_pieces() {
        let starts = |b: usize| b * 4;
        let scores = [0.25, 0.1, 0.9, 0.8, 0.5, 0.0];
        let entries = [(0, 4), (2, 4), (3, 2)];
        let record = reduce(
            &RawSelection {
                kv: 3,
                query_position: 24,
                scores: &scores,
                entries: &entries,
                block_start: &starts,
            },
            &pieces(),
            7,
        );
        assert_eq!(record.step, 7);
        assert_eq!(record.candidates, 6);
        assert_eq!(record.attended_blocks, 3);
        assert_eq!(record.attended_cells, 10);
        assert_eq!(record.cutoff, 0.25);
        assert_eq!(record.pieces_unread, 0, "c is listed, as a near miss");
        let row = |l: &str| record.pieces.iter().find(|p| p.label == l).unwrap();
        assert_eq!(row("b").attended_cells, 6);
        assert_eq!(row("b").best_rank, 0);
        assert_eq!(row("a").attended_cells, 4);
        assert_eq!(row("a").best_rank, 3);
        assert_eq!(row("c").attended_cells, 0);
        assert_eq!(row("c").attended_blocks, 0);
        assert_eq!(row("c").blocks, 2);
        assert_eq!(row("c").best_score, 0.5);
        assert_eq!(row("c").best_rank, 2);
        assert_eq!(record.top[0].position, 8);
        assert_eq!(record.top[0].piece.as_deref(), Some("b"));
        assert!(!record.top[2].attended);
    }

    /// A position past every piece belongs to none, and a hole between two
    /// belongs to neither.
    #[test]
    fn a_position_finds_its_piece() {
        let p = vec![
            PieceSpan {
                label: "a".into(),
                start: 4,
                end: 8,
            },
            PieceSpan {
                label: "b".into(),
                start: 10,
                end: 12,
            },
        ];
        assert_eq!(piece_of(&p, 3), None);
        assert_eq!(piece_of(&p, 4), Some(0));
        assert_eq!(piece_of(&p, 7), Some(0));
        assert_eq!(piece_of(&p, 8), None);
        assert_eq!(piece_of(&p, 11), Some(1));
        assert_eq!(piece_of(&p, 12), None);
    }

    /// One capture at a time, and disarming hands its records over once.
    #[test]
    fn a_capture_is_held_by_one_run() {
        let spec = Arm {
            control: Control::Dense,
            timeline: 9,
            from: 3,
            live_at: 8,
        };
        assert!(arm(spec));
        assert!(armed());
        assert!(dense());
        assert!(!fresh());
        assert_eq!(armed_with(), Some(spec));
        assert!(!arm(spec), "a second run is refused");
        set_step([5], 2, 1);
        assert!(watching(5));
        assert!(!watching(6));
        record_step(5, &[0.0, 2.0, 1.0], 1);
        record_step(6, &[0.0], 0);
        assert!(reproject_span().is_some());
        let capture = disarm();
        assert_eq!(capture.steps.len(), 1, "only a watched sequence records");
        assert_eq!(capture.reprojects.len(), 1);
        assert_eq!(capture.reprojects[0].step, 2);
        assert!(!armed());
        assert!(!dense());
        assert!(reproject_span().is_none());
    }

    /// The best ids rank by logit, ties to the lower id, and the digest is
    /// FNV-1a over the bits, on a value worked by hand.
    #[test]
    fn a_step_names_its_best_ids_and_digests_its_row() {
        let record = reduce_step(&[1.0, 3.0, 3.0, -1.0], 2, 4, 6);
        assert_eq!(record.top, vec![(1, 3.0), (2, 3.0), (0, 1.0), (3, -1.0)]);
        assert_eq!(record.sampled, 2);
        assert_eq!(record.rows, 6);
        assert_eq!(digest(&[]), 0xcbf2_9ce4_8422_2325);
        // (0xcbf29ce484222325 ^ 0x3f800000) * 0x100000001b3 mod 2^64.
        assert_eq!(digest(&[1.0]), 0x2f63_bdaa_d481_b7df);
        assert_ne!(digest(&[1.0, 3.0]), digest(&[3.0, 1.0]));
    }
}
