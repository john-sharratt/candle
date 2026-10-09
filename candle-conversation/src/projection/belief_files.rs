//! A belief turn group's candidates, assembled and scored: every file's
//! exchanges as gallery windows ([`assemble_file_scans`]), then any number of
//! probes scored against them in one pass ([`scan_file_scans`]) — one paged GPU
//! launch over the resident gallery arena, or the CPU per-file scan on a host
//! without one.
//!
//! Shared by the live belief scan (`Conversation::score_belief_groups`, one
//! probe plus its question window) and the ingest normalization warm-up, which
//! scores a file's own turns against it: assembling a file is host work
//! proportional to its turns, so the warm-up assembles it once for all of its
//! self-probes instead of once per probe.

use std::ops::Range;
use std::sync::Arc;

use super::ids::{TimelineId, TurnIndex};
use super::resolver::{sig_fingerprint, subwindow_bounds};
use super::schema::GroupSchema;
use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::streams::StreamId;
use crate::provenance::gallery_arena::{PagedSegment, PagedWindow};
use crate::provenance::heads_per_group;
use crate::provenance::{
    score_slots_grouped, score_slots_weighted, FusionMode, GalleryArena, WideQSig,
};
use crate::substrate::Substrate;
use crate::summary_tree::exchange::{exchanges, over_normals};

/// One file's candidates: its turns grouped into exchanges, and the gallery
/// windows that score them.
pub(crate) struct FileScan {
    pub timeline: TimelineId,
    /// Every non-summary turn, in order — the subsequence couplings project onto.
    pub arc_turn: Vec<TurnIndex>,
    /// Each exchange's span of `arc_turn`.
    pub ex_ranges: Vec<Range<usize>>,
    pub n_slots: usize,
    /// The signature windows referenced by `windows`, kept alive for the scan.
    pub arcs_kept: Vec<Arc<Vec<WideQSig>>>,
    /// Stream id of each kept arc's turn — the arena residency key.
    pub arc_sids: Vec<StreamId>,
    /// `(arc index into arcs_kept, window start, window end, exchange slot)`.
    pub windows: Vec<(usize, usize, usize, usize)>,
    /// Real signature tokens per exchange slot — the Concept A.4 size input.
    pub ex_tokens: Vec<usize>,
}

/// Assemble every file in `timelines` from `sub`. Returns the files that hold
/// at least one window, and how many turns were walked to find them.
pub(crate) fn assemble_file_scans(
    sub: &Substrate,
    timelines: Vec<TimelineId>,
) -> (Vec<FileScan>, usize) {
    let mut turns_walked = 0usize;
    let mut files: Vec<FileScan> = Vec::new();
    for timeline in timelines {
        // Enumerate the group's turns exactly as selection does — the whole
        // resolved timeline, `0..turn_count`, fetched by stream id — instead
        // of scanning `all_streams()` per group (which is O(all timelines'
        // streams) on the reproject hot path).
        let count = sub.turn_count(timeline);
        turns_walked += count as usize;
        // Per candidate turn: its full sig plus the self-referencing sub-window
        // seams recorded on it. A turn with no seams scores as one whole-turn
        // window; a turn with N seams scores as N+1 focused windows that all
        // resolve back to it — so a query matching one structural region of a
        // prefilled listing surfaces the whole turn without diluting against
        // the rest.
        let mut arcs: Vec<Option<Arc<Vec<WideQSig>>>> = Vec::new();
        let mut arc_turn: Vec<TurnIndex> = Vec::new();
        let mut arc_bounds: Vec<Vec<(usize, usize)>> = Vec::new();
        for i in 0..count {
            let idx = TurnIndex(i);
            // Summary forest nodes are selected only by the score-density
            // path, never the belief/rule path — mirror project.rs and skip
            // them so a summary can't take a raw turn's belief slot.
            if sub
                .tree_meta_of(timeline, idx)
                .map(|m| m.kind.is_summary())
                .unwrap_or(false)
            {
                continue;
            }
            // Keep EVERY non-summary (Normal) turn in `arc_turn`, so it is the
            // COMPLETE Normal subsequence the couplings project onto. A turn that
            // has no wide-Q sig (e.g. a prefilled tool-response half) must still
            // hold its position, or `over_normals`/`exchanges` would fuse a
            // coupled call with the wrong later turn. A sig-less turn contributes
            // no gallery window (empty bounds) but still joins its exchange.
            arc_turn.push(idx);
            match sub.decoded_wide_sig(turn_stream_id(timeline.raw(), i)) {
                Some(window) => {
                    // Self-referencing projection events mark sub-window seams.
                    // Read the memoized (sorted, deduped) seams — decoded from the
                    // events JSON once per session, not re-parsed for every gallery
                    // turn on every reprojection — and derive contiguous
                    // `[start, end)` bounds; no seams ⇒ one whole-turn window.
                    // `subwindow_bounds` ignores any seam at/past `len`, so the raw
                    // offsets are safe to pass straight through.
                    let len = window.len();
                    let seams = sub.decoded_seams(turn_stream_id(timeline.raw(), i));
                    arc_bounds.push(subwindow_bounds(len, &seams));
                    arcs.push(Some(window));
                }
                None => {
                    arc_bounds.push(Vec::new());
                    arcs.push(None);
                }
            }
        }
        if arc_turn.is_empty() {
            continue;
        }
        // Group the kept turns into EXCHANGES before scoring. A code-read scope
        // (and any tool round-trip) is a *coupled pair* — a `<tool_call>` turn
        // and its `<tool_response>` turn joined by a `TurnCoupling` record — and
        // provenance must hit the pair as ONE unit (`exchange_of`): scoring the
        // two halves as separate candidates splits the scope's vote and lets the
        // generic call framing (near-identical across every scope) compete on
        // its own. `arc_turn` is the COMPLETE Normal subsequence in chronological
        // order, so `over_normals` maps the call-turn indices straight onto arc
        // positions and the response is always the next position. Uncoupled turns
        // are their own singleton exchange.
        let couplings = over_normals(&sub.couplings_of(timeline), &arc_turn);
        let ex_ranges = exchanges(&couplings, arc_turn.len());
        let mut ex_slot = vec![0usize; arc_turn.len()];
        for (slot, r) in ex_ranges.iter().enumerate() {
            for ai in r.clone() {
                ex_slot[ai] = slot;
            }
        }
        let n_slots = ex_ranges.len();
        // Flatten every turn's sub-windows into gallery windows, all tagged with
        // their EXCHANGE slot (`wslot[i] = exchange index`). So the scan
        // aggregates a whole round-trip — every sub-window of the call AND the
        // response — into ONE case (best-token agreement across the pair),
        // rather than letting the halves (or a turn's own regions) compete as
        // separate cases and split the scope's vote. Then the L46-weighted vote
        // (§83) decides the exchange. Only the arcs actually referenced by a
        // window are kept alive.
        let mut arcs_kept: Vec<Arc<Vec<WideQSig>>> = Vec::new();
        let mut arc_sids: Vec<StreamId> = Vec::new();
        let mut windows: Vec<(usize, usize, usize, usize)> = Vec::new();
        for (ai, arc) in arcs.iter().enumerate() {
            let Some(arc) = arc else {
                continue; // sig-less turn: no window, but keeps its exchange slot
            };
            let sid = turn_stream_id(timeline.raw(), arc_turn[ai].0);
            let mut ki: Option<usize> = None;
            for &(s, e) in &arc_bounds[ai] {
                if e > s {
                    let k = *ki.get_or_insert_with(|| {
                        arcs_kept.push(arc.clone());
                        arc_sids.push(sid);
                        arcs_kept.len() - 1
                    });
                    windows.push((k, s, e, ex_slot[ai]));
                }
            }
        }
        if windows.is_empty() {
            continue;
        }
        let mut ex_tokens = vec![0usize; n_slots];
        for &(_, s, e, slot) in &windows {
            ex_tokens[slot] += e - s;
        }
        files.push(FileScan {
            timeline,
            arc_turn,
            ex_ranges,
            n_slots,
            arcs_kept,
            arc_sids,
            windows,
            ex_tokens,
        });
    }
    (files, turns_walked)
}

/// The score competitions `files` form, each as the indices of its files.
///
/// **A competition needs two cases.** The scorer votes `z × margin`, the
/// margin being the leading case's agreement over the runner-up's. A file of
/// one exchange has no runner-up, so its margin is its raw agreement — about
/// half the signature bits for any pair at all — and the file is scored as
/// itself against nothing. A corpus of one-exchange documents (a world's
/// entries, one era each) then ranks by size and genericity, normalization
/// compresses every document into the same band, and the same derelict ship
/// came top for every character whatever it was doing. So every file of one
/// exchange competes in one gallery, a case each, and a file of several
/// exchanges stays its own competition, its exchanges against each other.
pub(crate) fn competitions(files: &[FileScan]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut pooled: Vec<usize> = Vec::new();
    for (i, f) in files.iter().enumerate() {
        match f.n_slots {
            1 => pooled.push(i),
            _ => out.push(vec![i]),
        }
    }
    if !pooled.is_empty() {
        out.push(pooled);
    }
    out
}

/// One competition's gallery windows, the case each votes for, and its case
/// count: each file's exchanges at its offset among the competition's cases.
fn competition_windows<'a>(
    files: &'a [FileScan],
    comp: &[usize],
) -> (Vec<&'a [WideQSig]>, Vec<usize>, usize) {
    let mut wref = Vec::new();
    let mut wslot = Vec::new();
    let mut n_cases = 0usize;
    for &fi in comp {
        let f = &files[fi];
        for &(k, s, e, slot) in &f.windows {
            wref.push(&f.arcs_kept[k][s..e]);
            wslot.push(n_cases + slot);
        }
        n_cases += f.n_slots;
    }
    (wref, wslot, n_cases)
}

/// `v` at exactly `n` entries — a scorer that answered short is zeros beyond.
fn resized(mut v: Vec<f32>, n: usize) -> Vec<f32> {
    v.resize(n, 0.0);
    v
}

/// Split `votes` — every competition's cases laid end to end, in competition
/// order — back into one vector per file, in file order.
fn scatter(files: &[FileScan], comps: &[Vec<usize>], votes: &[f32]) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); files.len()];
    let mut at = 0usize;
    for comp in comps {
        for &fi in comp {
            let n = files[fi].n_slots;
            let end = (at + n).min(votes.len());
            out[fi] = resized(votes.get(at..end).unwrap_or(&[]).to_vec(), n);
            at += n;
        }
    }
    out
}

/// Score every probe against every file: `[probe][file]` of `(fused, mass
/// base)` — the policy-fused per-exchange scores and the UNGATED additive sum
/// Concept B reads. Every probe rides the same launch — one pass over the
/// gallery and one sync — each voting exactly as it would alone.
///
/// GPU = ONE paged segmented launch over the resident gallery arena (per-file
/// z / margin / needle gate, numerically equivalent to the CPU per-file scan up
/// to fast-math ULP — same ranking); CPU = the identical `score_slots_weighted`
/// per file, on a host without the arena or when the arena declines. Per-layer-
/// group vote weights come from the group's `policy.layer_weights` (empty ⇒
/// uniform).
pub(crate) fn scan_file_scans(
    files: &[FileScan],
    group: &GroupSchema,
    probes: &[&[WideQSig]],
    arena: Option<&GalleryArena>,
) -> Vec<Vec<(Vec<f32>, Vec<f32>)>> {
    let weights = &group.policy.layer_weights;
    let comps = competitions(files);
    // The CPU scan of one probe — the path a host without the arena takes —
    // one competition at a time, its votes laid end to end in competition
    // order, as the arena lays them.
    let scan_files_cpu = |p: &[WideQSig]| -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut fused_all: Vec<f32> = Vec::new();
        let mut base_all: Vec<f32> = Vec::new();
        for comp in &comps {
            let (wref, wslot, n_cases) = competition_windows(files, comp);
            let (fused, base) = match group.policy.scan.fusion {
                FusionMode::Additive => {
                    let v = score_slots_weighted(p, &wref, &wslot, n_cases, weights);
                    (v.clone(), v)
                }
                mode => {
                    let grouped = score_slots_grouped(p, &wref, &wslot, n_cases, weights);
                    if grouped.is_empty() {
                        (vec![0.0; n_cases], vec![0.0; n_cases])
                    } else {
                        (mode.fuse(&grouped), FusionMode::Additive.fuse(&grouped))
                    }
                }
            };
            fused_all.extend(resized(fused, n_cases));
            base_all.extend(resized(base, n_cases));
        }
        scatter(files, &comps, &fused_all)
            .into_iter()
            .zip(scatter(files, &comps, &base_all))
            .collect()
    };
    let gpu_scores: Option<Vec<Vec<(Vec<f32>, Vec<f32>)>>> = arena.and_then(|arena| {
        // Per-turn residency fingerprint: the decoded-sig `Arc` identity + a
        // content sample (first/last words). The memo serves a STABLE `Arc`
        // that changes identity only when a turn's blob is rewritten (and keeps
        // unchanged Arcs alive), so the pointer is a cheap content key — a seal
        // bumps it, a reprojection against unchanged turns does not. The word
        // sample is the ABA guard against an in-place re-seal / substrate reset
        // reusing an address. Keyed per turn, so a seal re-uploads only that
        // turn's pages; unchanged turns stay resident (no upload).
        // One segment per competition, each file's exchanges at its offset in
        // the competition's cases.
        let segments: Vec<PagedSegment> = comps
            .iter()
            .map(|comp| {
                let mut windows = Vec::new();
                let mut n_cases = 0usize;
                for &fi in comp {
                    let f = &files[fi];
                    windows.extend(f.windows.iter().map(|&(k, s, e, slot)| PagedWindow {
                        sid: f.arc_sids[k],
                        fingerprint: sig_fingerprint(&f.arcs_kept[k]),
                        turn: f.arcs_kept[k].as_slice(),
                        start: s,
                        end: e,
                        case: n_cases + slot,
                    }));
                    n_cases += f.n_slots;
                }
                PagedSegment { windows, n_cases }
            })
            .collect();
        // `[probe][file][slot]`.
        let arena_scan = |w: &[f32]| -> Option<Vec<Vec<Vec<f32>>>> {
            match arena.scan_weighted(&segments, probes, w) {
                Ok(out) => {
                    // One per-GLOBAL-case vote vector per probe, in segment
                    // (competition) order — split back per file.
                    Some(
                        out.into_iter()
                            .map(|votes| scatter(files, &comps, &votes))
                            .collect(),
                    )
                }
                // WARN, and on the crate-qualified target — both deliberate.
                //
                // The target must fall under one of the host's `EnvFilter`
                // directives (`zend=`, `candle_conversation=`,
                // `candle_transformers=`, `candle_nn=`); a bare `provenance`
                // matches none, so the fallback could never be printed and its
                // absence from a log would read as proof it had not happened.
                //
                // And on a CUDA host a declined GPU scan is not a debug detail:
                // the CPU per-file path is orders of magnitude slower over a
                // large corpus, so the symptom is a daemon that has quietly
                // become slow rather than one that reports a fault. The CPU path
                // stays — a host without CUDA has no other way to score beliefs —
                // but taking it on a machine that has a GPU is worth saying out
                // loud.
                Err(e) => {
                    tracing::warn!(
                        target: "candle_conversation::provenance",
                        "paged GPU belief scan unavailable, using CPU per-file scan: {e}"
                    );
                    None
                }
            }
        };
        // Concept G on the arena: non-additive modes run one one-hot scan per
        // fold group (each IS that group's needle-gated tally), then fuse per
        // the mode — the same law as the CPU `score_slots_fused`. The ungated
        // group sum rides along as the mass base.
        match group.policy.scan.fusion {
            FusionMode::Additive => arena_scan(weights).map(|per_probe| {
                per_probe
                    .into_iter()
                    .map(|per_file| per_file.into_iter().map(|v| (v.clone(), v)).collect())
                    .collect()
            }),
            mode => {
                // Derived from the probe's head count, not a locked constant.
                let n_groups = probes.first().and_then(|p| p.first()).map(|s| {
                    let n = s.n_heads as usize;
                    match heads_per_group(n) {
                        0 => 1,
                        per => (n / per).max(1),
                    }
                })?;
                // grouped[g][probe][file][slot]
                let mut grouped: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(n_groups);
                for g in 0..n_groups {
                    let mut one_hot = vec![0.0f32; n_groups];
                    one_hot[g] = weights.get(g).copied().unwrap_or(1.0);
                    grouped.push(arena_scan(&one_hot)?);
                }
                Some(
                    (0..probes.len())
                        .map(|pi| {
                            (0..files.len())
                                .map(|fi| {
                                    let per_group: Vec<Vec<f32>> =
                                        grouped.iter().map(|gp| gp[pi][fi].clone()).collect();
                                    (mode.fuse(&per_group), FusionMode::Additive.fuse(&per_group))
                                })
                                .collect()
                        })
                        .collect(),
                )
            }
        }
    });
    gpu_scores.unwrap_or_else(|| probes.iter().map(|p| scan_files_cpu(p)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random folded signature: 3 groups × 4 heads ×
    /// 2 words.
    fn sig(seed: u64) -> WideQSig {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let words = (0..24)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect();
        WideQSig { n_heads: 12, words }
    }

    /// A file of `exchanges` one-turn exchanges of `tokens` signature tokens
    /// each, seeded from `seed`.
    fn file(timeline: u64, seed: u64, exchanges: usize, tokens: usize) -> FileScan {
        let arcs: Vec<Arc<Vec<WideQSig>>> = (0..exchanges)
            .map(|e| {
                Arc::new(
                    (0..tokens)
                        .map(|t| sig(seed * 10_000 + (e * tokens + t) as u64))
                        .collect(),
                )
            })
            .collect();
        let tl = TimelineId::for_test(timeline);
        FileScan {
            timeline: tl,
            arc_turn: (0..exchanges as u32).map(TurnIndex).collect(),
            ex_ranges: (0..exchanges).map(|e| e..e + 1).collect(),
            n_slots: exchanges,
            arc_sids: (0..exchanges as u32)
                .map(|i| turn_stream_id(tl.raw(), i))
                .collect(),
            windows: (0..exchanges).map(|e| (e, 0, tokens, e)).collect(),
            ex_tokens: vec![tokens; exchanges],
            arcs_kept: arcs,
        }
    }

    /// Files of one exchange compete together, a file of several on its own.
    #[test]
    fn single_exchange_files_compete_as_one_gallery() {
        let files = vec![file(1, 1, 1, 4), file(2, 2, 3, 4), file(3, 3, 1, 4)];
        assert_eq!(competitions(&files), vec![vec![1], vec![0, 2]]);
        // Votes laid out as the competitions are — file 1's three, then the
        // pool's 0 and 2 — come back in file order.
        let votes = [10.0, 11.0, 12.0, 20.0, 30.0];
        assert_eq!(
            scatter(&files, &competitions(&files), &votes),
            vec![vec![20.0], vec![10.0, 11.0, 12.0], vec![30.0]]
        );
    }

    /// **The pooled document that holds the probe wins, and only it.** Scored
    /// alone, a one-exchange document has no runner-up, so its margin is its
    /// raw agreement and every document scores well against a probe that is
    /// none of them; pooled, the margin is over the next document and a
    /// document the probe is not drawn from scores next to nothing.
    #[test]
    fn a_pooled_document_is_scored_against_the_others() {
        let files: Vec<FileScan> = (1..=5).map(|i| file(i, i, 1, 24)).collect();
        // The probe is a stretch of document 3's own signature.
        let probe: Vec<WideQSig> = files[2].arcs_kept[0][4..16].to_vec();

        let comps = competitions(&files);
        assert_eq!(comps, vec![vec![0, 1, 2, 3, 4]]);
        let (wref, wslot, n_cases) = competition_windows(&files, &comps[0]);
        let pooled = scatter(
            &files,
            &comps,
            &score_slots_weighted(&probe, &wref, &wslot, n_cases, &[]),
        );
        let others = [0, 1, 3, 4].map(|i| pooled[i][0]);
        assert!(
            others.iter().all(|&o| pooled[2][0] > 10.0 * o.max(1e-3)),
            "pooled: {pooled:?}"
        );

        // Alone, a document the probe is not from still scores most of what
        // the right one does.
        let alone = |f: &FileScan| {
            let (wref, wslot, n) = competition_windows(std::slice::from_ref(f), &[0]);
            score_slots_weighted(&probe, &wref, &wslot, n, &[])[0]
        };
        assert!(alone(&files[0]) > 0.2 * alone(&files[2]));
    }
}
