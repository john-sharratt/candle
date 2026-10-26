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

use std::cell::OnceCell;
use std::ops::Range;

use super::adaptive::ScanCases;
use super::ids::{TimelineId, TurnIndex};
use super::resolver::subwindow_bounds;
use super::schema::GroupSchema;
use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::streams::StreamId;
use crate::provenance::gallery_arena::{PagedSegment, PagedWindow};
use crate::provenance::heads_per_group;
use crate::provenance::{
    score_slots_grouped, score_slots_weighted, FusionMode, GalleryArena, SigWindow, WideQSig,
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
    /// The stored signature windows referenced by `windows`, not decoded.
    pub arcs_kept: Vec<SigWindow>,
    /// Stream id of each kept arc's turn — the arena residency key.
    pub arc_sids: Vec<StreamId>,
    /// `(arc index into arcs_kept, window start, window end, exchange slot)`:
    /// the file's own windows.
    pub windows: Vec<(usize, usize, usize, usize)>,
    /// `(arc index into arcs_kept, window start, window end)`: each question
    /// its calibrations hold ([`Substrate::CALIBRATES_KEY`]), a window apiece.
    /// They vote in the calibration competition, not among the file's own
    /// windows ([`scan_file_scans`]).
    pub cal_windows: Vec<(usize, usize, usize)>,
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
        let mut arcs: Vec<Option<SigWindow>> = Vec::new();
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
            match sub.wide_sig_window(turn_stream_id(timeline.raw(), i)) {
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
        let mut arcs_kept: Vec<SigWindow> = Vec::new();
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
        // Each calibration's questions, a window apiece, for the calibration
        // competition. Counted in no exchange's size: they are not its content.
        let mut cal_windows: Vec<(usize, usize, usize)> = Vec::new();
        for cal in sub.calibrations_of(timeline) {
            for i in 0..sub.turn_count(cal) {
                let sid = turn_stream_id(cal.raw(), i);
                let Some(window) = sub.wide_sig_window(sid) else {
                    continue;
                };
                let len = window.len();
                if len == 0 {
                    continue;
                }
                arcs_kept.push(window);
                arc_sids.push(sid);
                cal_windows.push((arcs_kept.len() - 1, 0, len));
            }
        }
        files.push(FileScan {
            timeline,
            arc_turn,
            ex_ranges,
            n_slots,
            arcs_kept,
            arc_sids,
            windows,
            cal_windows,
            ex_tokens,
        });
    }
    (files, turns_walked)
}

/// The score competitions `files` form under `cases`, each as the indices of
/// its files.
///
/// **A competition needs two cases.** The scorer votes `z × margin`, the
/// margin being the leading case's agreement over the runner-up's. A file of
/// one exchange has no runner-up, so its margin is its raw agreement — about
/// half the signature bits for any pair at all — and the file is scored as
/// itself against nothing. A corpus of one-exchange documents (a world's
/// entries, one era each) then ranks by size and genericity, normalization
/// compresses every document into the same band, and the same derelict ship
/// came top for every character whatever it was doing. So under
/// [`ScanCases::Exchange`] every file of one exchange competes in one gallery,
/// a case each, and a file of several exchanges stays its own competition, its
/// exchanges against each other. Under [`ScanCases::File`] every file competes
/// in the one gallery, a case each, however many exchanges it has.
pub(crate) fn competitions(files: &[FileScan], cases: ScanCases) -> Vec<Vec<usize>> {
    if cases == ScanCases::File {
        return if files.is_empty() {
            Vec::new()
        } else {
            vec![(0..files.len()).collect()]
        };
    }
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

/// How many cases `f` is under `cases`.
fn n_cases_of(f: &FileScan, cases: ScanCases) -> usize {
    match cases {
        ScanCases::Exchange => f.n_slots,
        ScanCases::File => 1,
    }
}

/// The case exchange `slot` of a file votes for under `cases`, within the
/// file's own cases.
fn case_of(slot: usize, cases: ScanCases) -> usize {
    match cases {
        ScanCases::Exchange => slot,
        ScanCases::File => 0,
    }
}

/// Every file's kept windows decoded, `[file][arc]` — what the host scan
/// reads, held only for the scan that needs it.
fn decode_files(files: &[FileScan]) -> Vec<Vec<Vec<WideQSig>>> {
    files
        .iter()
        .map(|f| f.arcs_kept.iter().map(SigWindow::decode).collect())
        .collect()
}

/// One competition's gallery windows, the case each votes for, and its case
/// count: each file's exchanges at its offset among the competition's cases.
/// `decoded` is [`decode_files`] of `files`.
fn competition_windows<'a>(
    files: &[FileScan],
    decoded: &'a [Vec<Vec<WideQSig>>],
    comp: &[usize],
    cases: ScanCases,
) -> (Vec<&'a [WideQSig]>, Vec<usize>, usize) {
    let mut wref = Vec::new();
    let mut wslot = Vec::new();
    let mut n_cases = 0usize;
    for &fi in comp {
        let f = &files[fi];
        for &(k, s, e, slot) in &f.windows {
            wref.push(&decoded[fi][k][s..e]);
            wslot.push(n_cases + case_of(slot, cases));
        }
        n_cases += n_cases_of(f, cases);
    }
    (wref, wslot, n_cases)
}

/// `v` at exactly `n` entries — a scorer that answered short is zeros beyond.
fn resized(mut v: Vec<f32>, n: usize) -> Vec<f32> {
    v.resize(n, 0.0);
    v
}

/// Split `votes` — every competition's cases laid end to end, in competition
/// order — back into one vector per file, one score per exchange, in file
/// order. Under [`ScanCases::File`] a file's one score stands for each of its
/// exchanges.
fn scatter(
    files: &[FileScan],
    comps: &[Vec<usize>],
    votes: &[f32],
    cases: ScanCases,
) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); files.len()];
    let mut at = 0usize;
    for comp in comps {
        for &fi in comp {
            let f = &files[fi];
            let n = n_cases_of(f, cases);
            let end = (at + n).min(votes.len());
            let own = resized(votes.get(at..end).unwrap_or(&[]).to_vec(), n);
            out[fi] = match cases {
                ScanCases::Exchange => own,
                ScanCases::File => vec![own[0]; f.n_slots],
            };
            at += n;
        }
    }
    out
}

/// The calibration competition: every file that holds a calibration window, a
/// case each, in file order.
fn calibrated(files: &[FileScan]) -> Vec<usize> {
    files
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.cal_windows.is_empty())
        .map(|(i, _)| i)
        .collect()
}

/// The calibration competition's windows and the case each votes for:
/// `cal[i]`'s calibration windows vote for case `i`. `decoded` is
/// [`decode_files`] of `files`.
fn calibration_windows<'a>(
    files: &[FileScan],
    decoded: &'a [Vec<Vec<WideQSig>>],
    cal: &[usize],
) -> (Vec<&'a [WideQSig]>, Vec<usize>) {
    let mut wref = Vec::new();
    let mut wslot = Vec::new();
    for (case, &fi) in cal.iter().enumerate() {
        for &(k, s, e) in &files[fi].cal_windows {
            wref.push(&decoded[fi][k][s..e]);
            wslot.push(case);
        }
    }
    (wref, wslot)
}

/// Split `votes` — every file competition's cases laid end to end, then the
/// calibration competition's, one case per file of `cal` — into one vector
/// per file ([`scatter`]), each calibrated file's calibration vote weighed by
/// `weight` and added to its head exchange; under [`ScanCases::File`], to the
/// file's one score, which every exchange carries.
fn with_calibration(
    files: &[FileScan],
    comps: &[Vec<usize>],
    cal: &[usize],
    votes: &[f32],
    cases: ScanCases,
    weight: f32,
) -> Vec<Vec<f32>> {
    let own: usize = files.iter().map(|f| n_cases_of(f, cases)).sum();
    let mut out = scatter(files, comps, &votes[..own.min(votes.len())], cases);
    for (i, &fi) in cal.iter().enumerate() {
        let v = weight * votes.get(own + i).copied().unwrap_or(0.0);
        match cases {
            ScanCases::Exchange => {
                if let Some(head) = out[fi].first_mut() {
                    *head += v;
                }
            }
            ScanCases::File => out[fi].iter_mut().for_each(|x| *x += v),
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
///
/// **Calibrations compete among themselves.** A calibration's exemplars and a
/// live probe share a distribution, so in one pool with the files' content
/// they take nearly every vote, whatever the content says — measured, 96.5%
/// of a background's vote mass at 2% of the gallery, and question recall
/// below the content's alone. Every calibrated file is a case of one more
/// competition, scored on its calibration windows alone with its own `z` and
/// margin, in the same launch; its vote, weighed by the policy's
/// `calibration_weight`, is added to the file's own score.
pub(crate) fn scan_file_scans(
    files: &[FileScan],
    group: &GroupSchema,
    probes: &[&[WideQSig]],
    arena: Option<&GalleryArena>,
) -> Vec<Vec<(Vec<f32>, Vec<f32>)>> {
    let weights = &group.policy.layer_weights;
    let cases = group.policy.scan.cases;
    let cal_weight = group.policy.scan.calibration_weight;
    let comps = competitions(files, cases);
    let cal = calibrated(files);
    // The CPU scan of one probe — the path a host without the arena takes —
    // one competition at a time, its votes laid end to end in competition
    // order, as the arena lays them. The windows are decoded on the host path's
    // first use and shared by every probe of this call.
    let decoded: OnceCell<Vec<Vec<Vec<WideQSig>>>> = OnceCell::new();
    let scan_files_cpu = |p: &[WideQSig]| -> Vec<(Vec<f32>, Vec<f32>)> {
        let decoded = decoded.get_or_init(|| decode_files(files));
        let mut fused_all: Vec<f32> = Vec::new();
        let mut base_all: Vec<f32> = Vec::new();
        let mut score = |wref: &[&[WideQSig]], wslot: &[usize], n_cases: usize| {
            let (fused, base) = match group.policy.scan.fusion {
                FusionMode::Additive => {
                    let v = score_slots_weighted(p, wref, wslot, n_cases, weights);
                    (v.clone(), v)
                }
                mode => {
                    let grouped = score_slots_grouped(p, wref, wslot, n_cases, weights);
                    if grouped.is_empty() {
                        (vec![0.0; n_cases], vec![0.0; n_cases])
                    } else {
                        (mode.fuse(&grouped), FusionMode::Additive.fuse(&grouped))
                    }
                }
            };
            fused_all.extend(resized(fused, n_cases));
            base_all.extend(resized(base, n_cases));
        };
        for comp in &comps {
            let (wref, wslot, n_cases) = competition_windows(files, decoded, comp, cases);
            score(&wref, &wslot, n_cases);
        }
        if !cal.is_empty() {
            let (wref, wslot) = calibration_windows(files, decoded, &cal);
            score(&wref, &wslot, cal.len());
        }
        let split = |votes: &[f32]| with_calibration(files, &comps, &cal, votes, cases, cal_weight);
        split(&fused_all)
            .into_iter()
            .zip(split(&base_all))
            .collect()
    };
    let gpu_scores: Option<Vec<Vec<(Vec<f32>, Vec<f32>)>>> = arena.and_then(|arena| {
        // Per-turn residency fingerprint: the stored window's
        // (`SigWindow::fingerprint`), which changes only when the turn's blob is
        // rewritten — so a seal re-uploads only that turn's pages, unchanged
        // turns stay resident, and a resident turn is never decoded.
        // One segment per competition, each file's exchanges at its offset in
        // the competition's cases.
        let mut segments: Vec<PagedSegment> = comps
            .iter()
            .map(|comp| {
                let mut windows = Vec::new();
                let mut n_cases = 0usize;
                for &fi in comp {
                    let f = &files[fi];
                    windows.extend(f.windows.iter().map(|&(k, s, e, slot)| PagedWindow {
                        sid: f.arc_sids[k],
                        fingerprint: f.arcs_kept[k].fingerprint(),
                        turn: (&f.arcs_kept[k]).into(),
                        start: s,
                        end: e,
                        case: n_cases + case_of(slot, cases),
                    }));
                    n_cases += n_cases_of(f, cases);
                }
                PagedSegment { windows, n_cases }
            })
            .collect();
        // The calibration competition, last: each calibrated file a case.
        if !cal.is_empty() {
            let mut windows = Vec::new();
            for (case, &fi) in cal.iter().enumerate() {
                let f = &files[fi];
                windows.extend(f.cal_windows.iter().map(|&(k, s, e)| PagedWindow {
                    sid: f.arc_sids[k],
                    fingerprint: f.arcs_kept[k].fingerprint(),
                    turn: (&f.arcs_kept[k]).into(),
                    start: s,
                    end: e,
                    case,
                }));
            }
            segments.push(PagedSegment {
                windows,
                n_cases: cal.len(),
            });
        }
        // `[probe][file][slot]`.
        let arena_scan = |w: &[f32]| -> Option<Vec<Vec<Vec<f32>>>> {
            match arena.scan_weighted(&segments, probes, w) {
                Ok(out) => {
                    // One per-GLOBAL-case vote vector per probe, in segment
                    // (competition) order — split back per file, the
                    // calibration competition's added on.
                    Some(
                        out.into_iter()
                            .map(|votes| {
                                with_calibration(files, &comps, &cal, &votes, cases, cal_weight)
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::projection::Builder;
    use crate::provenance::encode_wide_sigs;

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
        let arcs: Vec<SigWindow> = (0..exchanges)
            .map(|e| {
                let sigs: Vec<WideQSig> = (0..tokens)
                    .map(|t| sig(seed * 10_000 + (e * tokens + t) as u64))
                    .collect();
                SigWindow::new(Arc::from(encode_wide_sigs(&sigs))).expect("a stored window")
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
            cal_windows: Vec::new(),
            ex_tokens: vec![tokens; exchanges],
            arcs_kept: arcs,
        }
    }

    /// A group scored under `cases`, every other policy at its default.
    fn schema_group(cases: ScanCases) -> GroupSchema {
        let cases = match cases {
            ScanCases::Exchange => "exchange",
            ScanCases::File => "file",
        };
        let yaml = format!(
            r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: files
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: scopes
        selection: {{ kind: top_k, k: 2 }}
        policy: {{ cases: {cases} }}
"#
        );
        Builder::from_yaml(&yaml).unwrap().schema().layers[0].groups[0].clone()
    }

    /// `file`, with a calibration of `questions` windows, one per question.
    fn calibrated_file(
        timeline: u64,
        seed: u64,
        exchanges: usize,
        tokens: usize,
        questions: &[Vec<WideQSig>],
    ) -> FileScan {
        let mut f = file(timeline, seed, exchanges, tokens);
        for (i, q) in questions.iter().enumerate() {
            f.arcs_kept
                .push(SigWindow::new(Arc::from(encode_wide_sigs(q))).expect("a stored window"));
            f.arc_sids.push(turn_stream_id(9_000 + timeline, i as u32));
            f.cal_windows.push((f.arcs_kept.len() - 1, 0, q.len()));
        }
        f
    }

    /// **The calibration competition's votes land on their files.** Votes
    /// laid out as the launch lays them — the file competitions' cases, then
    /// one per calibrated file — come back with each calibrated file's
    /// calibration vote, weighed, on its head exchange; by file, on its one
    /// score.
    #[test]
    fn a_calibration_vote_is_weighed_onto_its_file() {
        let q = vec![vec![sig(1)]];
        let files = vec![
            file(1, 1, 1, 4),
            calibrated_file(2, 2, 3, 4, &q),
            calibrated_file(3, 3, 1, 4, &q),
        ];
        assert_eq!(calibrated(&files), vec![1, 2]);
        let ex = ScanCases::Exchange;
        let comps = competitions(&files, ex);
        // File 1's three exchanges, the pool's 0 and 2, then the calibration
        // competition's 1 and 2.
        let votes = [10.0, 11.0, 12.0, 20.0, 30.0, 4.0, 8.0];
        assert_eq!(
            with_calibration(&files, &comps, &[1, 2], &votes, ex, 0.5),
            vec![vec![20.0], vec![12.0, 11.0, 12.0], vec![34.0]]
        );
        let by_file = ScanCases::File;
        let comps = competitions(&files, by_file);
        assert_eq!(
            with_calibration(
                &files,
                &comps,
                &[1, 2],
                &[20.0, 10.0, 30.0, 4.0, 8.0],
                by_file,
                0.25
            ),
            vec![vec![20.0], vec![11.0, 11.0, 11.0], vec![32.0]]
        );
    }

    /// **Questions compete among themselves.** Three files, each calibrated
    /// with a question; a probe drawn from the second's question, in a pool
    /// of content that matches none of them, ranks the second first through
    /// the calibration competition alone — and with a weight of 0 it would
    /// not be told apart.
    #[test]
    fn a_probe_like_a_calibration_question_finds_its_file() {
        let questions: Vec<Vec<WideQSig>> = (0..3)
            .map(|i| (0..12).map(|t| sig(50_000 + i * 100 + t)).collect())
            .collect();
        let files: Vec<FileScan> = (0..3)
            .map(|i| calibrated_file(i + 1, i + 1, 1, 24, &questions[i as usize..i as usize + 1]))
            .collect();
        let probe: Vec<WideQSig> = questions[1][2..10].to_vec();
        let mut group = schema_group(ScanCases::File);
        let scores = |g: &GroupSchema| {
            scan_file_scans(&files, g, &[probe.as_slice()], None)[0]
                .iter()
                .map(|(fused, _)| fused[0])
                .collect::<Vec<f32>>()
        };
        let weighed = scores(&group);
        assert!(
            weighed[1] > weighed[0] && weighed[1] > weighed[2],
            "{weighed:?}"
        );
        group.policy.scan.calibration_weight = 0.0;
        let unweighed = scores(&group);
        assert!(unweighed[1] < weighed[1], "{unweighed:?} vs {weighed:?}");
    }

    /// Files of one exchange compete together, a file of several on its own.
    #[test]
    fn single_exchange_files_compete_as_one_gallery() {
        let files = vec![file(1, 1, 1, 4), file(2, 2, 3, 4), file(3, 3, 1, 4)];
        let ex = ScanCases::Exchange;
        assert_eq!(competitions(&files, ex), vec![vec![1], vec![0, 2]]);
        // Votes laid out as the competitions are — file 1's three, then the
        // pool's 0 and 2 — come back in file order.
        let votes = [10.0, 11.0, 12.0, 20.0, 30.0];
        assert_eq!(
            scatter(&files, &competitions(&files, ex), &votes, ex),
            vec![vec![20.0], vec![10.0, 11.0, 12.0], vec![30.0]]
        );
    }

    /// **By file, every file competes in one gallery, a case each.** A file
    /// of several exchanges is one case, all its windows voting for it, and
    /// its one score stands for each of its exchanges.
    #[test]
    fn by_file_every_file_is_one_case_of_one_gallery() {
        let files = vec![file(1, 1, 1, 4), file(2, 2, 3, 4), file(3, 3, 1, 4)];
        let by_file = ScanCases::File;
        let comps = competitions(&files, by_file);
        assert_eq!(comps, vec![vec![0, 1, 2]]);
        let decoded = decode_files(&files);
        let (_, wslot, n_cases) = competition_windows(&files, &decoded, &comps[0], by_file);
        assert_eq!(n_cases, 3);
        assert_eq!(wslot, vec![0, 1, 1, 1, 2]);
        assert_eq!(
            scatter(&files, &comps, &[20.0, 10.0, 30.0], by_file),
            vec![vec![20.0], vec![10.0, 10.0, 10.0], vec![30.0]]
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
        let probe: Vec<WideQSig> = files[2].arcs_kept[0].decode()[4..16].to_vec();

        let ex = ScanCases::Exchange;
        let comps = competitions(&files, ex);
        assert_eq!(comps, vec![vec![0, 1, 2, 3, 4]]);
        let decoded = decode_files(&files);
        let (wref, wslot, n_cases) = competition_windows(&files, &decoded, &comps[0], ex);
        let pooled = scatter(
            &files,
            &comps,
            &score_slots_weighted(&probe, &wref, &wslot, n_cases, &[]),
            ex,
        );
        let others = [0, 1, 3, 4].map(|i| pooled[i][0]);
        assert!(
            others.iter().all(|&o| pooled[2][0] > 10.0 * o.max(1e-3)),
            "pooled: {pooled:?}"
        );

        // Alone, a document the probe is not from still scores most of what
        // the right one does.
        let alone = |f: &FileScan| {
            let one = std::slice::from_ref(f);
            let decoded = decode_files(one);
            let (wref, wslot, n) = competition_windows(one, &decoded, &[0], ex);
            score_slots_weighted(&probe, &wref, &wslot, n, &[])[0]
        };
        assert!(alone(&files[0]) > 0.2 * alone(&files[2]));
    }
}
