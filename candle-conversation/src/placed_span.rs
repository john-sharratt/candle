//! Runs of a turn's user half whose K/V is placed from another turn rather than
//! prefilled (`docs/inline_read_injection.md`).
//!
//! A caller names a run by its bytes in the half and the conversation that holds
//! the same text ([`PlacedText`]); at submit the conversation finds the run in
//! one of that conversation's turns and cuts the part to place on that turn's
//! grid ([`place`]), and the scheduler stops the prefill at the span's start
//! ([`snap_advance`]), places the source's K/V and index pages, and prefills on
//! from the span's end.

use std::ops::Range;

use candle_nn::CHUNK_SIZE;

use crate::projection::{TimelineId, TurnIndex};
use crate::sealed_turn::{SealedPages, SealedTurn};

/// Ids left off each end of a run before it is looked for: a multi-call round
/// labels each block (`[i/n file_read]`), and the label can merge with the
/// body's first token, so a run's outermost ids need not be the source's.
pub const SLACK: usize = 8;

/// A run of a user half to place from `source`: its bytes in the half, and how
/// many grid steps of each edge to prefill ([`place`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedText {
    pub bytes: Range<usize>,
    pub source: TimelineId,
    pub edge: usize,
}

/// A resolved placement: the prefill positions the placed tokens occupy, and
/// where the same tokens sit in the source turn's grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedSpan {
    pub tokens: Range<usize>,
    pub source: TimelineId,
    pub turn: TurnIndex,
    pub source_tokens: Range<usize>,
}

/// Where in one of `turns` the run `run` — a user half's ids from its position
/// `at` on — is placed from: the run's interior found in exactly one turn, and
/// cut from the `edge`-th grid point past its start to the `edge`-th before its
/// end. The grid is every index page's start, each [`CHUNK_SIZE`] tokens into a
/// page, and every page's end; a turn with no pages is one page.
///
/// The span is returned with `source` left for the caller to fill.
pub fn place(
    run: &[u32],
    at: usize,
    turns: &[SealedTurn],
    edge: usize,
) -> Result<(Range<usize>, u32, Range<usize>), String> {
    if edge == 0 {
        return Err("a placed run keeps at least one grid step of each edge".to_string());
    }
    if run.len() <= 2 * SLACK {
        return Err(format!(
            "a run of {} ids has no interior to find",
            run.len()
        ));
    }
    let core = &run[SLACK..run.len() - SLACK];
    let mut found = turns.iter().flat_map(|t| {
        occurrences(&t.token_ids, core)
            .into_iter()
            .map(move |s| (t, s))
    });
    let Some((turn, s)) = found.next() else {
        return Err(format!(
            "the run's {} interior ids occur in none of the source's {} turns",
            core.len(),
            turns.len()
        ));
    };
    if found.next().is_some() {
        return Err("the run's interior occurs more than once in the source".to_string());
    }
    let grid = grid(&turn.pages, turn.token_ids.len())?;
    let e = s + core.len();
    let start = grid.iter().copied().filter(|&g| g > s).nth(edge - 1);
    let end = grid.iter().copied().rev().filter(|&g| g < e).nth(edge - 1);
    let (Some(start), Some(end)) = (start, end) else {
        return Err("the run is too short to leave a placed range between its edges".to_string());
    };
    if start >= end {
        return Err("the run is too short to leave a placed range between its edges".to_string());
    }
    // Position `x` of the source grid is the run's `SLACK + (x - s)`.
    let live = |x: usize| at + SLACK + (x - s);
    Ok((live(start)..live(end), turn.index, start..end))
}

/// Every position at which `needle` starts in `hay`.
fn occurrences(hay: &[u32], needle: &[u32]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return Vec::new();
    }
    hay.windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i)
        .collect()
}

/// The grid points of a turn of `len` tokens whose index is `pages`, ascending.
fn grid(pages: &SealedPages, len: usize) -> Result<Vec<usize>, String> {
    let widths = match pages {
        SealedPages::Widths(w) => w.clone(),
        SealedPages::Absent => vec![len],
        SealedPages::Malformed(e) => return Err(format!("the source turn's pages: {e}")),
    };
    let mut points = Vec::new();
    let mut start = 0usize;
    for w in widths {
        let end = start + w;
        points.extend((start..end).step_by(CHUNK_SIZE));
        start = end;
    }
    points.push(start);
    points.dedup();
    Ok(points)
}

/// How far a prefill at `offset` may advance, of the `advance` it was given,
/// so that it stops at the start of every placed span.
pub fn snap_advance(spans: &[PlacedSpan], offset: usize, advance: usize) -> usize {
    spans
        .iter()
        .filter(|s| offset < s.tokens.start && offset + advance > s.tokens.start)
        .map(|s| s.tokens.start - offset)
        .min()
        .unwrap_or(advance)
}

/// The placed span starting at `offset`, if one does.
pub fn span_at(spans: &[PlacedSpan], offset: usize) -> Option<&PlacedSpan> {
    spans.iter().find(|s| s.tokens.start == offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(index: u32, ids: Vec<u32>, pages: SealedPages) -> SealedTurn {
        SealedTurn {
            index,
            token_ids: ids,
            pages,
        }
    }

    /// Ids 1000.. — none of them small, so a run of them is unambiguous.
    fn ids(n: usize) -> Vec<u32> {
        (1000..1000 + n as u32).collect()
    }

    /// A run read from positions 70..200 of a turn whose pages are [100, 200].
    /// Its interior is 78..192 and the grid is 0 32 64 96 | 100 132 164 196 228
    /// 260 | 300, so the second point past 78 is the page start 100 and the
    /// second before 192 is 132, a chunk into the second page.
    #[test]
    fn a_run_is_found_and_cut_on_the_source_grid() {
        let source = ids(300);
        let run = source[70..200].to_vec();
        let turns = [
            turn(0, vec![7; 50], SealedPages::Absent),
            turn(3, source, SealedPages::Widths(vec![100, 200])),
        ];
        let (tokens, index, from) = place(&run, 10, &turns, 2).unwrap();
        assert_eq!(index, 3);
        assert_eq!(from, 100..132);
        // The run's id at source position x sits at 10 + (x - 70).
        assert_eq!(tokens, 40..72);
        assert_eq!(
            run[tokens.start - 10..tokens.end - 10],
            turns[1].token_ids[100..132]
        );
    }

    #[test]
    fn a_run_found_nowhere_or_twice_is_refused() {
        let run = ids(40);
        let absent = [turn(0, vec![1; 100], SealedPages::Absent)];
        assert!(place(&run, 0, &absent, 1)
            .unwrap_err()
            .contains("occur in none"));
        let twice = [
            turn(0, run.clone(), SealedPages::Absent),
            turn(1, run.clone(), SealedPages::Absent),
        ];
        assert_eq!(
            place(&run, 0, &twice, 1).unwrap_err(),
            "the run's interior occurs more than once in the source"
        );
    }

    /// Interior 18..102 on the grid 0 32 64 96 128 160 192 200: two steps of
    /// each edge meet at 64, one step leaves 32..96.
    #[test]
    fn a_run_too_short_for_its_edges_is_refused() {
        let source = ids(200);
        let run = source[10..110].to_vec();
        let turns = [turn(0, source, SealedPages::Absent)];
        assert_eq!(
            place(&run, 0, &turns, 2).unwrap_err(),
            "the run is too short to leave a placed range between its edges"
        );
        assert_eq!(place(&run, 0, &turns, 1).unwrap().2, 32..96);
    }

    #[test]
    fn a_chunk_stops_at_a_placed_span() {
        let spans = [PlacedSpan {
            tokens: 40..104,
            source: TimelineId::for_test(1),
            turn: TurnIndex(3),
            source_tokens: 100..164,
        }];
        assert_eq!(snap_advance(&spans, 0, 512), 40);
        assert_eq!(snap_advance(&spans, 0, 40), 40, "already stops there");
        assert_eq!(snap_advance(&spans, 104, 512), 512, "past it");
        assert!(span_at(&spans, 40).is_some());
        assert!(span_at(&spans, 41).is_none());
    }
}
