//! Which pieces of a slot a rebuild keeps, which it splices in or out where
//! they stand, and where it cuts the slot and walks the rest.
//!
//! A projection may insert and evict anywhere: provenance picks the pieces, and
//! the order they are placed in is the projection's, not the history's. A
//! rebuild that kept only the longest prefix it shared with the slot re-placed
//! every piece behind the first one that changed — on a codebase tour, about
//! eighty turns every sixty-four tokens. Here the slot's pieces are aligned
//! with the projection's by each piece's own identity, so an unchanged piece
//! stays where it is wherever the change falls, and only what changed is
//! spliced.
//!
//! **What can stand still, and what cannot.** A sealed piece — a section, a
//! turn, a turn's half — is K/V computed once and borrowed, and the RoPE base
//! of every block is re-derived from the slot's layout, so it can sit anywhere.
//! A forwarded piece — glue, the live user message — is K/V this slot computed
//! against everything before it, so a change anywhere before it changes it; its
//! identity is chained for exactly that reason, and the first forwarded piece
//! that does not match is where the slot is cut and the rest walked.
//!
//! **Where the model allows a boundary.** Per-position state may be grouped
//! more coarsely than the pieces — one page over a slot's whole injected
//! prefix — so every splice and the cut must land on a position the model can
//! cut at. A plan that cannot is shortened to the last one that can.

use std::ops::Range;

/// What a piece's K/V is to the slot holding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PieceKind {
    /// Borrowed as sealed: it may move, and be spliced in or out, anywhere.
    Sealed,
    /// Computed by this slot against what precedes it: kept only while
    /// everything before it is.
    Forwarded,
}

/// A piece the slot holds: what it is, and the position it ends at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HeldPiece {
    pub identity: u64,
    pub kind: PieceKind,
    pub end_pos: usize,
}

/// A piece the projection wants, in the order it wants it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WantedPiece {
    pub identity: u64,
    pub kind: PieceKind,
}

/// Held pieces `held` give way to wanted pieces `wanted`, where the held ones
/// stood.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PieceEdit {
    pub held: Range<usize>,
    pub wanted: Range<usize>,
}

/// How a rebuild turns the slot's pieces into the projection's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SplicePlan {
    /// Held pieces kept through the cut: the slot is cut after the first
    /// `cut_after` of them, and every held piece after is let go.
    pub cut_after: usize,
    /// Splices inside what the cut keeps, front to back. Applied back to
    /// front, each one's positions are those of the slot as it stood.
    pub edits: Vec<PieceEdit>,
    /// Wanted pieces from here on are walked onto the slot's tail.
    pub walk_from: usize,
    /// For each wanted piece before `walk_from`, the held piece it keeps —
    /// `None` for one an edit puts in.
    pub keeps: Vec<Option<usize>>,
}

impl SplicePlan {
    /// The plan that keeps nothing: cut to empty and walk everything.
    fn walk_all() -> Self {
        Self {
            cut_after: 0,
            edits: Vec::new(),
            walk_from: 0,
            keeps: Vec::new(),
        }
    }

    /// Held pieces the plan keeps where they stand.
    pub fn kept(&self) -> usize {
        self.keeps.iter().filter(|k| k.is_some()).count()
    }

    /// Wanted pieces the plan's edits put in.
    pub fn inserted(&self) -> usize {
        self.edits.iter().map(|e| e.wanted.len()).sum()
    }

    /// Held pieces the plan's edits take out.
    pub fn evicted(&self) -> usize {
        self.edits.iter().map(|e| e.held.len()).sum()
    }
}

/// Plan the rebuild of a slot holding `held` into one holding `wanted`.
///
/// `floor(pos)` is the largest position at or before `pos` the model can cut
/// its per-position state at — `pos` itself where nothing is grouped across it.
pub(super) fn plan_splice<E>(
    held: &[HeldPiece],
    wanted: &[WantedPiece],
    floor: impl Fn(usize) -> Result<usize, E>,
) -> Result<SplicePlan, E> {
    let mut pairs = matched_pairs(held, wanted);
    // The first forwarded piece the slot does not already hold is computed
    // against everything before it, so nothing from there on can be kept.
    let first_new_forwarded = wanted
        .iter()
        .enumerate()
        .filter(|(_, w)| w.kind == PieceKind::Forwarded)
        .map(|(j, _)| j)
        .find(|j| !pairs.iter().any(|&(_, w)| w == *j))
        .unwrap_or(wanted.len());
    pairs.retain(|&(_, w)| w < first_new_forwarded);
    if pairs.is_empty() {
        return Ok(SplicePlan::walk_all());
    }

    let start_pos = |i: usize| if i == 0 { 0 } else { held[i - 1].end_pos };
    let exact = |pos: usize| floor(pos).map(|f| f == pos);
    // The edit in front of each pair, and whether the model can make it.
    let mut edits: Vec<PieceEdit> = Vec::with_capacity(pairs.len());
    let mut usable = 0usize;
    let mut prev = (0usize, 0usize);
    for (k, &(h, w)) in pairs.iter().enumerate() {
        let edit = PieceEdit {
            held: prev.0..h,
            wanted: prev.1..w,
        };
        let aligned = edit.held.is_empty() && edit.wanted.is_empty()
            || (exact(start_pos(edit.held.start))? && exact(start_pos(edit.held.end))?);
        if !aligned {
            break;
        }
        edits.push(edit);
        if exact(held[h].end_pos)? {
            usable = k + 1;
        }
        prev = (h + 1, w + 1);
    }
    if usable == 0 {
        return Ok(SplicePlan::walk_all());
    }
    let (last_held, last_wanted) = pairs[usable - 1];
    edits.truncate(usable);
    let mut keeps = vec![None; last_wanted + 1];
    for &(h, w) in &pairs[..usable] {
        keeps[w] = Some(h);
    }
    edits.retain(|e| !(e.held.is_empty() && e.wanted.is_empty()));
    Ok(SplicePlan {
        cut_after: last_held + 1,
        edits,
        walk_from: last_wanted + 1,
        keeps,
    })
}

/// The longest common subsequence of the two lists by identity, as increasing
/// `(held, wanted)` index pairs.
fn matched_pairs(held: &[HeldPiece], wanted: &[WantedPiece]) -> Vec<(usize, usize)> {
    let (n, m) = (held.len(), wanted.len());
    // `len[i][j]`: the longest match of `held[i..]` with `wanted[j..]`.
    let mut len = vec![0u32; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            len[at(i, j)] = if held[i].identity == wanted[j].identity {
                len[at(i + 1, j + 1)] + 1
            } else {
                len[at(i + 1, j)].max(len[at(i, j + 1)])
            };
        }
    }
    let mut pairs = Vec::with_capacity(len[at(0, 0)] as usize);
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if held[i].identity == wanted[j].identity {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if len[at(i + 1, j)] >= len[at(i, j + 1)] {
            i += 1;
        } else {
            j += 1;
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Held pieces of these identities, each `width` tokens — sealed unless
    /// named in `forwarded`.
    fn held(ids: &[u64], width: usize, forwarded: &[u64]) -> Vec<HeldPiece> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| HeldPiece {
                identity: id,
                kind: kind(id, forwarded),
                end_pos: (i + 1) * width,
            })
            .collect()
    }

    fn wanted(ids: &[u64], forwarded: &[u64]) -> Vec<WantedPiece> {
        ids.iter()
            .map(|&id| WantedPiece {
                identity: id,
                kind: kind(id, forwarded),
            })
            .collect()
    }

    fn kind(id: u64, forwarded: &[u64]) -> PieceKind {
        if forwarded.contains(&id) {
            PieceKind::Forwarded
        } else {
            PieceKind::Sealed
        }
    }

    /// Every position is a boundary the model can cut at.
    fn anywhere(pos: usize) -> Result<usize, ()> {
        Ok(pos)
    }

    fn plan(h: &[HeldPiece], w: &[WantedPiece]) -> SplicePlan {
        plan_splice(h, w, anywhere).expect("plan")
    }

    /// **The case the splice exists for.** A section near the front flips and
    /// eighty turns stand behind it: the rebuild replaces that one piece where
    /// it stands and keeps every turn, instead of cutting at the flip and
    /// re-placing them all.
    #[test]
    fn a_change_near_the_front_is_spliced_and_everything_behind_it_kept() {
        let mut ids: Vec<u64> = vec![1, 2, 3];
        ids.extend(100..180);
        let mut now = ids.clone();
        now[1] = 9;
        let p = plan(&held(&ids, 10, &[]), &wanted(&now, &[]));
        assert_eq!(
            p.edits,
            [PieceEdit {
                held: 1..2,
                wanted: 1..2
            }]
        );
        assert_eq!(p.cut_after, ids.len());
        assert_eq!(p.walk_from, now.len());
        assert_eq!(p.kept(), ids.len() - 1);
        assert_eq!((p.inserted(), p.evicted()), (1, 1));
    }

    /// Insert, evict and replace in one rebuild, each where it falls, and a
    /// new piece past the last held one is walked onto the tail.
    #[test]
    fn inserts_evictions_and_replacements_land_where_they_fall() {
        let p = plan(
            &held(&[1, 2, 3, 4, 5], 10, &[]),
            &wanted(&[1, 7, 2, 4, 8, 5, 6], &[]),
        );
        assert_eq!(
            p.edits,
            [
                PieceEdit {
                    held: 1..1,
                    wanted: 1..2
                },
                PieceEdit {
                    held: 2..3,
                    wanted: 3..3
                },
                PieceEdit {
                    held: 4..4,
                    wanted: 4..5
                },
            ]
        );
        assert_eq!(p.cut_after, 5);
        assert_eq!(p.walk_from, 6, "6 is past every held piece");
        assert_eq!(
            p.keeps,
            [Some(0), None, Some(1), Some(3), None, Some(4)],
            "each kept wanted piece names the held piece it is"
        );
    }

    /// Held pieces past the last one still wanted are let go by the cut.
    #[test]
    fn held_pieces_past_the_last_kept_one_are_cut() {
        let p = plan(&held(&[1, 2, 3, 4], 10, &[]), &wanted(&[1, 2, 5], &[]));
        assert_eq!(p.cut_after, 2);
        assert!(p.edits.is_empty());
        assert_eq!(p.walk_from, 2);
    }

    /// A forwarded piece the slot does not hold is computed against everything
    /// before it, so the walk starts there: nothing after it is kept, though a
    /// held piece after it is still wanted.
    #[test]
    fn the_first_new_forwarded_piece_starts_the_walk() {
        let p = plan(
            &held(&[1, 2, 3, 4], 10, &[]),
            &wanted(&[1, 50, 3, 4], &[50]),
        );
        assert_eq!(p.cut_after, 1);
        assert!(p.edits.is_empty());
        assert_eq!(p.walk_from, 1, "the glue at 1 and all after it are walked");
    }

    /// A forwarded piece the slot already holds is kept: its chained identity
    /// matching says everything before it is unchanged.
    #[test]
    fn a_held_forwarded_piece_is_kept() {
        let p = plan(
            &held(&[1, 50, 2], 10, &[50]),
            &wanted(&[1, 50, 2, 3], &[50]),
        );
        assert_eq!((p.cut_after, p.walk_from), (3, 3));
        assert!(p.edits.is_empty());
    }

    /// The model groups 0..30 into one page: an edit inside the group cannot
    /// be made, so the plan stops before it and walks from there — and a cut
    /// inside the group falls back to nothing kept.
    #[test]
    fn an_edit_the_model_cannot_cut_at_shortens_the_plan() {
        let grouped =
            |pos: usize| -> Result<usize, ()> { Ok(if pos > 0 && pos < 30 { 0 } else { pos }) };
        let h = held(&[1, 2, 3, 4, 5], 10, &[]);
        // 2 → 9 sits inside 0..30: nothing before it can be cut at either.
        let p = plan_splice(&h, &wanted(&[1, 9, 3, 4, 5], &[]), grouped).unwrap();
        assert_eq!(p, SplicePlan::walk_all());
        // 4 → 9 sits at 30..40, past the group: spliced, all else kept.
        let p = plan_splice(&h, &wanted(&[1, 2, 3, 9, 5], &[]), grouped).unwrap();
        assert_eq!(
            p.edits,
            [PieceEdit {
                held: 3..4,
                wanted: 3..4
            }]
        );
        assert_eq!((p.cut_after, p.walk_from), (5, 5));
    }

    /// The case that failed live: a dialogue base whose whole injected prefix
    /// is one page, 0..1338, and a question whose projection first differs
    /// after the 3-token opening. Nothing can be cut or spliced inside the
    /// page, so nothing is kept.
    #[test]
    fn a_change_inside_a_page_over_the_prefix_keeps_nothing() {
        let one_page =
            |pos: usize| -> Result<usize, ()> { Ok(if pos > 0 && pos < 1338 { 0 } else { pos }) };
        let ends = [3usize, 120, 1338, 1400];
        let h: Vec<HeldPiece> = [1u64, 2, 3, 4]
            .iter()
            .zip(ends)
            .map(|(&identity, end_pos)| HeldPiece {
                identity,
                kind: PieceKind::Sealed,
                end_pos,
            })
            .collect();
        let p = plan_splice(&h, &wanted(&[1, 9, 3, 4], &[]), one_page).unwrap();
        assert_eq!(p, SplicePlan::walk_all());
        // A change after the page: cut at the page's end, keeping all of it,
        // and the new last piece walked onto the tail.
        let p = plan_splice(&h, &wanted(&[1, 2, 3, 9], &[]), one_page).unwrap();
        assert!(p.edits.is_empty());
        assert_eq!((p.cut_after, p.walk_from, p.kept()), (3, 3, 3));
    }

    /// An edit after a misaligned one is never applied: the plan keeps the
    /// pairs before the first edit it cannot make, and cuts only where the
    /// model can — here 10..30 is one group, so the cut lands at 10.
    #[test]
    fn the_plan_stops_at_the_first_edit_it_cannot_make() {
        let no_cut_in_10_30 =
            |pos: usize| -> Result<usize, ()> { Ok(if pos > 10 && pos < 30 { 10 } else { pos }) };
        let h = held(&[1, 2, 3, 4, 5], 10, &[]);
        let p = plan_splice(&h, &wanted(&[1, 2, 9, 4, 8], &[]), no_cut_in_10_30).unwrap();
        assert_eq!((p.cut_after, p.walk_from), (1, 1));
        assert!(p.edits.is_empty());
    }

    /// Nothing in common walks everything; identical lists keep everything.
    #[test]
    fn nothing_shared_walks_all_and_everything_shared_keeps_all() {
        assert_eq!(
            plan(&held(&[1, 2], 10, &[]), &wanted(&[3, 4], &[])),
            SplicePlan::walk_all()
        );
        let same = plan(&held(&[1, 2, 3], 10, &[]), &wanted(&[1, 2, 3], &[]));
        assert_eq!((same.cut_after, same.walk_from, same.kept()), (3, 3, 3));
        assert!(same.edits.is_empty());
    }

    #[test]
    fn a_failing_floor_is_returned() {
        let h = held(&[1, 2], 10, &[]);
        assert_eq!(
            plan_splice(&h, &wanted(&[1, 2], &[]), |_| Err("lock poisoned")),
            Err("lock poisoned")
        );
    }
}
