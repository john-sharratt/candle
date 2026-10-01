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
    // The CPU per-file scan of one probe — the path a host without the arena
    // takes.
    let scan_files_cpu = |p: &[WideQSig]| -> Vec<(Vec<f32>, Vec<f32>)> {
        files
            .iter()
            .map(|f| {
                let wref: Vec<&[WideQSig]> = f
                    .windows
                    .iter()
                    .map(|&(k, s, e, _)| &f.arcs_kept[k][s..e])
                    .collect();
                let wslot: Vec<usize> = f.windows.iter().map(|&(_, _, _, slot)| slot).collect();
                match group.policy.scan.fusion {
                    FusionMode::Additive => {
                        let v = score_slots_weighted(p, &wref, &wslot, f.n_slots, weights);
                        (v.clone(), v)
                    }
                    mode => {
                        let grouped = score_slots_grouped(p, &wref, &wslot, f.n_slots, weights);
                        if grouped.is_empty() {
                            (vec![0.0; f.n_slots], vec![0.0; f.n_slots])
                        } else {
                            (mode.fuse(&grouped), FusionMode::Additive.fuse(&grouped))
                        }
                    }
                }
            })
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
        let segments: Vec<PagedSegment> = files
            .iter()
            .map(|f| PagedSegment {
                windows: f
                    .windows
                    .iter()
                    .map(|&(k, s, e, slot)| PagedWindow {
                        sid: f.arc_sids[k],
                        fingerprint: sig_fingerprint(&f.arcs_kept[k]),
                        turn: f.arcs_kept[k].as_slice(),
                        start: s,
                        end: e,
                        case: slot,
                    })
                    .collect(),
                n_cases: f.n_slots,
            })
            .collect();
        // `[probe][file][slot]`.
        let arena_scan = |w: &[f32]| -> Option<Vec<Vec<Vec<f32>>>> {
            match arena.scan_weighted(&segments, probes, w) {
                Ok(out) => {
                    // One per-GLOBAL-case vote vector per probe, in segment
                    // (file) order. Split each back per file by cumulative
                    // `n_slots`.
                    Some(
                        out.into_iter()
                            .map(|votes| {
                                let mut cum = 0usize;
                                files
                                    .iter()
                                    .map(|f| {
                                        let end = (cum + f.n_slots).min(votes.len());
                                        let mut v = votes.get(cum..end).unwrap_or(&[]).to_vec();
                                        v.resize(f.n_slots, 0.0);
                                        cum += f.n_slots;
                                        v
                                    })
                                    .collect()
                            })
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
