//! `tool_select_from_substrate` — model-free tool-selection harness.
//!
//! Loads the real substrate (no model), rebuilds the 93 tool sections' signature
//! corpus, scans probe Q-signatures against it, and reports whether the right
//! tool is selected. Probes come from one of two sources, and there are two
//! per-token analysis modes:
//!
//! ```bash
//! # default: legacy fixture probes (single-tool-context prompts) — eval the
//! # production formula + sweep statistic×depth-weight formulas by accuracy.
//! cargo run -p zend --example tool_select_from_substrate --release -- <workspace>
//!
//! # --turns [user|asst]: probes from the clean 2-turn captures in the substrate,
//! # scanning the user-prompt half or the assistant tool-call half of each turn.
//! cargo run ... -- <workspace> --turns asst
//!
//! # --diag <tool>: per-token agreement of one captured assistant turn vs every
//! # tool section (raw, masked, leaderboard, consecutive n-gram) — locates which
//! # token/depth carries the signal instead of averaging it away.
//! cargo run ... -- <workspace> --diag weather
//!
//! # --score2 <L> <window>: aggregate Top-1/Top-5/MRR of consecutive L-gram
//! # matching over the assistant name-region window, across all captured turns.
//! cargo run ... -- <workspace> --score2 3 12
//! ```
//!
//! See `docs/tool_selection_provenance_ideas.md` for the broader plan these
//! diagnostic modes feed.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_conversation::models::Model;
use candle_conversation::persistence::resume::decode_signatures;
use candle_conversation::persistence::streams::{StreamDecl, StreamId, TurnDecl};
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::{
    Builder, DepthWeights, ProjectionMode, ProjectionTarget, ScoreFormula, SectionId,
    TimelineAllocator,
};
use candle_conversation::provenance::BdpScanner;
use candle_conversation::substrate::{PerDepthScores, ScoredSubstrate, Substrate, TurnScores};
use candle_conversation::{ProvenanceFile, SigEntry, TokenSignature};
use serde::{Deserialize, Serialize};
use zend::tools::install_tool_catalog;

const ZEND_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// Minimal mirror of the real-data manifest (only the fields we read).
#[derive(Deserialize)]
struct Manifest {
    scenarios: Vec<Scenario>,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    tool: Option<String>,
    #[serde(default)]
    case_type: String,
    #[serde(default)]
    user_prompt: String,
    byte_offset: u64,
    token_count: u16,
}

// ── The scoring formula: the optimisation target. Defaults mirror the
// production `tools` collection (FIXED_FORMULA `Span{2.0}`, prag-only weights,
// threshold 140.70, top-5). Edit `production()` or extend the sweep to tune.
#[derive(Clone)]
struct Formula {
    // scanner knobs (affect which statistics the scan computes)
    span_alpha: f32,
    scan_top_k: usize,
    hit_threshold: u32,
    // scoring knobs
    formula: ScoreFormula,
    weights: DepthWeights,
    // selection knobs
    score_threshold: f32,
    select_k: usize,
}

impl Formula {
    fn production() -> Self {
        Self {
            span_alpha: 2.0,
            scan_top_k: 8,
            hit_threshold: 90,
            formula: ScoreFormula::Span { alpha: 2.0 },
            weights: DepthWeights {
                syntactic: 0.0,
                semantic: 0.0,
                pragmatic: 1.0,
            },
            score_threshold: 140.70,
            select_k: 5,
        }
    }
}

/// The exact per-depth combine the projection uses (`combine_per_depth`):
/// a depth-weighted mean of the chosen statistic.
fn combined(pds: &PerDepthScores, f: ScoreFormula, w: &DepthWeights) -> f32 {
    w.combine(pds.syn.pick(f), pds.sem.pick(f), pds.prag.pick(f))
}

/// All tools ranked by combined score, highest first.
fn rank_tools<'a>(
    scores: &HashMap<SectionId, PerDepthScores>,
    sid_to_name: &HashMap<SectionId, &'a str>,
    f: ScoreFormula,
    w: &DepthWeights,
) -> Vec<(&'a str, f32)> {
    let mut v: Vec<(&str, f32)> = scores
        .iter()
        .filter_map(|(sid, p)| Some((*sid_to_name.get(sid)?, combined(p, f, w))))
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

// ── Capture structs (serialised to tool_select_debug.json for debugging).
#[derive(Serialize)]
struct Stats {
    max: f32,
    sum: f32,
    mean: f32,
    top_k_mean: f32,
    count: f32,
    span: f32,
    pertok_excess: f32,
}
impl Stats {
    fn of(t: &TurnScores) -> Self {
        Self {
            max: t.max,
            sum: t.sum,
            mean: t.mean,
            top_k_mean: t.top_k_mean,
            count: t.count,
            span: t.span,
            pertok_excess: t.pertok_excess,
        }
    }
}

#[derive(Serialize)]
struct Depths {
    syn: Stats,
    sem: Stats,
    prag: Stats,
}
impl Depths {
    fn of(p: &PerDepthScores) -> Self {
        Self {
            syn: Stats::of(&p.syn),
            sem: Stats::of(&p.sem),
            prag: Stats::of(&p.prag),
        }
    }
}

#[derive(Serialize)]
struct Ranked {
    tool: String,
    score: f32,
}

#[derive(Serialize)]
struct ProbeCapture {
    id: String,
    prompt: String,
    expected: String,
    expected_rank: usize, // 1-based; tools.len()+1 if absent
    expected_score: f32,
    passed_threshold: bool,
    selected: Vec<String>,            // manual: score ≥ threshold, top select_k
    correct: bool,                    // expected ∈ selected
    projection_selected: Vec<String>, // the real Decode projection (cross-check)
    top: Vec<Ranked>,                 // top-10 by score
    expected_breakdown: Depths,       // every statistic × depth for the expected tool
    top_distractor: Option<Ranked>,   // highest-ranked wrong tool
    top_distractor_breakdown: Option<Depths>,
}

#[derive(Serialize)]
struct Summary {
    n: usize,
    top1: f32,
    top_k: f32,
    mrr: f32,
    threshold_pass: f32,
    mean_rank: f32,
}

#[derive(Serialize)]
struct ConfigDump {
    span_alpha: f32,
    scan_top_k: usize,
    hit_threshold: u32,
    formula: String,
    weights: [f32; 3],
    score_threshold: f32,
    select_k: usize,
}

#[derive(Serialize)]
struct Report {
    config: ConfigDump,
    summary: Summary,
    probes: Vec<ProbeCapture>,
}

/// One scanned probe: the cached per-tool scores plus its identity.
struct Probe {
    id: String,
    prompt: String,
    expected: String,
    scores: HashMap<SectionId, PerDepthScores>,
    proj_scores: candle_conversation::substrate::ProjectionScores,
}

/// A `test_config.json` row — only the fields needed to map a captured turn to
/// its expected (canonical) tool by the verbatim user prompt.
#[derive(Deserialize)]
struct TestCase {
    tool: String,
    prompt: String,
}

/// Build probes from the captured conversation **turns** in the substrate (our
/// clean 2-turn dataset) rather than the fixture. `half` selects which side of
/// each turn to scan: `"user"` (the prompt) or `"asst"` (the tool-call reply).
/// A turn is mapped to its expected tool by matching its verbatim `user_text`
/// against `prompt_to_tool`, so only the tool-case captures are picked up
/// (titler / repo_map / code-read turns don't match and are skipped).
#[allow(clippy::too_many_arguments)]
fn build_turn_probes(
    substrate: &Substrate,
    persistence: &mut SubstratePersistence,
    provenance: &ProvenanceFile,
    corpus: &[(SectionId, Vec<SigEntry>)],
    cfg: &Formula,
    half: &str,
    prompt_to_tool: &HashMap<String, String>,
    valid_tools: &HashMap<String, SectionId>,
) -> anyhow::Result<Vec<Probe>> {
    let bl = TokenSignature::BYTE_LEN;
    let parse = |s: &[u8]| -> Vec<TokenSignature> {
        s.chunks_exact(bl)
            .map(|c| TokenSignature::from_bytes(c.try_into().unwrap()))
            .collect()
    };
    // Snapshot the turn streams up front so we don't hold a substrate borrow
    // across the signature reads.
    let turn_streams: Vec<(StreamId, TurnDecl)> = substrate
        .all_streams()
        .filter_map(|(sid, entry)| match &entry.decl {
            Some(StreamDecl::Turn(d)) => Some((sid, d.clone())),
            _ => None,
        })
        .collect();

    let mut probes = Vec::new();
    let mut diag_printed = false;
    for (stream_id, d) in turn_streams {
        let Some(tool) = prompt_to_tool.get(&d.user_text) else {
            continue; // not one of our tool-case captures
        };
        if !valid_tools.contains_key(tool) {
            continue;
        }
        let Some(payload) = persistence.read_signatures(substrate, stream_id)? else {
            continue;
        };
        let (mut syn_all, mut sem_all, mut prag_all) = (Vec::new(), Vec::new(), Vec::new());
        for (tc, bytes) in decode_signatures(&payload)? {
            let n = tc as usize;
            if bytes.len() < 3 * n * bl {
                continue;
            }
            syn_all.extend(parse(&bytes[0..n * bl]));
            sem_all.extend(parse(&bytes[n * bl..2 * n * bl]));
            prag_all.extend(parse(&bytes[2 * n * bl..3 * n * bl]));
        }
        let nsig = syn_all.len();
        if nsig == 0 {
            continue;
        }
        let (us, ue, ass) = (
            d.user_content_start as usize,
            d.user_content_end as usize,
            d.assistant_content_start as usize,
        );
        // If the sig array spans the whole grid, slice by content bounds; if it
        // only covers the decoded assistant tail, the whole array IS that half.
        let full_grid = nsig >= ass;
        if !diag_printed {
            eprintln!(
                "turn-probe coverage check: sigs={nsig} bounds(user {us}..{ue}, asst {ass}..) full_grid={full_grid}"
            );
            diag_printed = true;
        }
        let (lo, hi) = match half {
            "user" => {
                if full_grid && ue > us {
                    (us, ue.min(nsig))
                } else {
                    continue; // user-half sigs unavailable in this capture
                }
            }
            _ => {
                if full_grid {
                    (ass.min(nsig), nsig)
                } else {
                    (0, nsig)
                }
            }
        };
        if hi <= lo {
            continue;
        }
        let mut scanner = BdpScanner::new()
            .with_span_alpha(cfg.span_alpha)
            .with_top_k(cfg.scan_top_k)
            .with_hit_threshold(cfg.hit_threshold);
        scanner.scan_sections(
            provenance,
            &syn_all[lo..hi],
            &sem_all[lo..hi],
            &prag_all[lo..hi],
            corpus,
        )?;
        let scores: HashMap<SectionId, PerDepthScores> = corpus
            .iter()
            .filter_map(|(sid, _)| Some((*sid, *scanner.section_scores().get(sid)?)))
            .collect();
        probes.push(Probe {
            id: format!("{tool}/{half}"),
            prompt: d.user_text.clone(),
            expected: tool.clone(),
            scores,
            proj_scores: scanner.to_projection_scores(),
        });
    }
    Ok(probes)
}

/// Measure consecutive L-gram matching over the assistant name-region window
/// across every captured turn. For each turn, score each tool by the best
/// L-gram agreement-sum (max over probe start × section start), combining the
/// three depths by taking the best; rank tools; tally top-1 / top-5 / MRR.
#[allow(clippy::too_many_arguments)]
fn score_ngram(
    substrate: &Substrate,
    persistence: &mut SubstratePersistence,
    section_sigs: &HashMap<SectionId, [Vec<TokenSignature>; 3]>,
    name_to_sid: &HashMap<String, SectionId>,
    prompt_to_tool: &HashMap<String, String>,
    l: usize,
    window: usize,
) -> anyhow::Result<()> {
    let bl = TokenSignature::BYTE_LEN;
    let parse = |s: &[u8]| -> Vec<TokenSignature> {
        s.chunks_exact(bl)
            .map(|c| TokenSignature::from_bytes(c.try_into().unwrap()))
            .collect()
    };
    // Pre-convert section sigs to u128 per depth once.
    let sec_u128: HashMap<SectionId, [Vec<u128>; 3]> = section_sigs
        .iter()
        .map(|(sid, arr)| {
            (
                *sid,
                [
                    arr[0].iter().map(|s| s.as_u128()).collect(),
                    arr[1].iter().map(|s| s.as_u128()).collect(),
                    arr[2].iter().map(|s| s.as_u128()).collect(),
                ],
            )
        })
        .collect();

    let turn_streams: Vec<(StreamId, TurnDecl)> = substrate
        .all_streams()
        .filter_map(|(sid, e)| match &e.decl {
            Some(StreamDecl::Turn(d)) => Some((sid, d.clone())),
            _ => None,
        })
        .collect();

    let (mut n, mut top1, mut top5) = (0usize, 0usize, 0usize);
    let mut mrr = 0.0f64;
    for (stream_id, d) in turn_streams {
        let Some(tool) = prompt_to_tool.get(&d.user_text) else {
            continue;
        };
        let Some(&correct_sid) = name_to_sid.get(tool) else {
            continue;
        };
        let Some(payload) = persistence.read_signatures(substrate, stream_id)? else {
            continue;
        };
        let (mut sy, mut se, mut pr) = (Vec::new(), Vec::new(), Vec::new());
        for (tc, bytes) in decode_signatures(&payload)? {
            let nn = tc as usize;
            if bytes.len() < 3 * nn * bl {
                continue;
            }
            sy.extend(parse(&bytes[0..nn * bl]));
            se.extend(parse(&bytes[nn * bl..2 * nn * bl]));
            pr.extend(parse(&bytes[2 * nn * bl..3 * nn * bl]));
        }
        let nsig = sy.len();
        let ass = (d.assistant_content_start as usize).min(nsig);
        let hi = (ass + window).min(nsig);
        if hi <= ass {
            continue;
        }
        // Probe name-region window, per depth, as u128.
        let probe: [Vec<u128>; 3] = [
            sy[ass..hi].iter().map(|s| s.as_u128()).collect(),
            se[ass..hi].iter().map(|s| s.as_u128()).collect(),
            pr[ass..hi].iter().map(|s| s.as_u128()).collect(),
        ];
        // Score each tool: best L-gram agreement, best over the three depths.
        let mut scored: Vec<(SectionId, u32)> = sec_u128
            .iter()
            .map(|(sid, sarr)| {
                let mut best = 0u32;
                for depth in 0..3 {
                    let p = &probe[depth];
                    let s = &sarr[depth];
                    if p.len() < l || s.len() < l {
                        continue;
                    }
                    for i in 0..=p.len() - l {
                        for j in 0..=s.len() - l {
                            let mut sum = 0u32;
                            for k in 0..l {
                                sum += (!(p[i + k] ^ s[j + k])).count_ones();
                            }
                            best = best.max(sum);
                        }
                    }
                }
                (*sid, best)
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        let rank = scored
            .iter()
            .position(|(s, _)| *s == correct_sid)
            .map_or(usize::MAX, |x| x + 1);
        if rank != usize::MAX {
            n += 1;
            if rank == 1 {
                top1 += 1;
            }
            if rank <= 5 {
                top5 += 1;
            }
            mrr += 1.0 / rank as f64;
        }
    }
    let pct = |x: usize| 100.0 * x as f64 / n.max(1) as f64;
    println!(
        "\nn-gram L={l} window={window}: n={n}  Top-1 {:.1}%  Top-5 {:.1}%  MRR {:.3}  (chance Top-1 ≈1.1%, Top-5 ≈5.4%)",
        pct(top1),
        pct(top5),
        mrr / n.max(1) as f64
    );
    Ok(())
}

/// Per-token diagnostic: for the first captured turn of `target_tool`, scan each
/// assistant-half token against every tool section (max agreement over each
/// tool's tokens, per depth) and print the correct tool's rank per token. This
/// shows whether a *specific* token (e.g. the tool name) carries the signal that
/// the whole-half aggregate floods out.
fn run_diag(
    substrate: &Substrate,
    persistence: &mut SubstratePersistence,
    section_sigs: &HashMap<SectionId, [Vec<TokenSignature>; 3]>,
    name_to_sid: &HashMap<String, SectionId>,
    target_tool: &str,
    prompt_to_tool: &HashMap<String, String>,
) -> anyhow::Result<()> {
    let bl = TokenSignature::BYTE_LEN;
    let parse = |s: &[u8]| -> Vec<TokenSignature> {
        s.chunks_exact(bl)
            .map(|c| TokenSignature::from_bytes(c.try_into().unwrap()))
            .collect()
    };
    let correct_sid = *name_to_sid
        .get(target_tool)
        .ok_or_else(|| anyhow::anyhow!("unknown tool {target_tool}"))?;
    let turn = substrate
        .all_streams()
        .filter_map(|(sid, e)| match &e.decl {
            Some(StreamDecl::Turn(d)) => Some((sid, d.clone())),
            _ => None,
        })
        .find(|(_, d)| prompt_to_tool.get(&d.user_text).map(String::as_str) == Some(target_tool));
    let Some((stream_id, d)) = turn else {
        anyhow::bail!("no captured turn for tool {target_tool}");
    };
    let payload = persistence
        .read_signatures(substrate, stream_id)?
        .ok_or_else(|| anyhow::anyhow!("no sigs for that turn"))?;
    let (mut syn, mut sem, mut prag) = (Vec::new(), Vec::new(), Vec::new());
    for (tc, bytes) in decode_signatures(&payload)? {
        let n = tc as usize;
        if bytes.len() < 3 * n * bl {
            continue;
        }
        syn.extend(parse(&bytes[0..n * bl]));
        sem.extend(parse(&bytes[n * bl..2 * n * bl]));
        prag.extend(parse(&bytes[2 * n * bl..3 * n * bl]));
    }
    let nsig = syn.len();
    let ass = (d.assistant_content_start as usize).min(nsig);
    let depths: [(&str, &Vec<TokenSignature>, usize); 3] =
        [("syn", &syn, 0), ("sem", &sem, 1), ("prag", &prag, 2)];

    eprintln!("diag tool={target_tool}");
    eprintln!("  user_text     = {:?}", d.user_text);
    eprintln!("  assistant_text= {:?}", d.assistant_text);
    eprintln!(
        "  sigs={nsig}  assistant half = tokens {ass}..{nsig}  ({} tools in corpus)",
        section_sigs.len()
    );
    // Common-mode removal: raw agreements cluster ~80/128 because the sign
    // vectors share dominant directions (JSON/tool-format structure). Mask to the
    // bits whose sign actually VARIES across the corpus (population 20–80%) so
    // agreement counts only discriminative bits. Per depth.
    let total_tokens: usize = section_sigs.values().map(|a| a[0].len()).sum();
    let mut masks = [0u128; 3];
    for (d, mask) in masks.iter_mut().enumerate() {
        let mut ones = [0u32; 128];
        for arr in section_sigs.values() {
            for s in &arr[d] {
                let v = s.as_u128();
                for (b, o) in ones.iter_mut().enumerate() {
                    *o += ((v >> b) & 1) as u32;
                }
            }
        }
        let n = total_tokens as f32;
        for (b, &o) in ones.iter().enumerate() {
            let frac = o as f32 / n;
            if frac > 0.2 && frac < 0.8 {
                *mask |= 1u128 << b;
            }
        }
    }
    eprintln!(
        "  informative bits per depth: syn={} sem={} prag={} (of 128)",
        masks[0].count_ones(),
        masks[1].count_ones(),
        masks[2].count_ones()
    );

    // Rank of the correct tool by max agreement over a section's tokens, raw
    // (all 128 bits) vs masked (discriminative bits only).
    let rank_of = |probe: &TokenSignature, d: usize, mask: u128| -> (usize, usize) {
        let pv = probe.as_u128();
        let mut raw: Vec<(SectionId, u32)> = Vec::new();
        let mut msk: Vec<(SectionId, u32)> = Vec::new();
        for (sid, arr) in section_sigs {
            let (mut br, mut bm) = (0u32, 0u32);
            for s in &arr[d] {
                let x = !(pv ^ s.as_u128());
                br = br.max(x.count_ones());
                bm = bm.max((x & mask).count_ones());
            }
            raw.push((*sid, br));
            msk.push((*sid, bm));
        }
        raw.sort_by(|a, b| b.1.cmp(&a.1));
        msk.sort_by(|a, b| b.1.cmp(&a.1));
        let rr = raw
            .iter()
            .position(|(s, _)| *s == correct_sid)
            .map_or(0, |x| x + 1);
        let mr = msk
            .iter()
            .position(|(s, _)| *s == correct_sid)
            .map_or(0, |x| x + 1);
        (rr, mr)
    };

    println!(
        "\n{:>4}  {:<12} {:<12} {:<12}   (rawRank->maskedRank of correct tool; chance≈47)",
        "tok", "syn", "sem", "prag"
    );
    let mut best_masked: Option<(usize, &str, usize)> = None;
    for i in ass..nsig {
        let mut cols = Vec::new();
        for (dname, dvec, didx) in &depths {
            let (rr, mr) = rank_of(&dvec[i], *didx, masks[*didx]);
            cols.push(format!("{rr:>2}->{mr:<2}"));
            if mr > 0 && best_masked.map(|(_, _, b)| mr < b).unwrap_or(true) {
                best_masked = Some((i, dname, mr));
            }
        }
        println!("{:>4}  {:<12} {:<12} {:<12}", i, cols[0], cols[1], cols[2]);
    }
    match best_masked {
        Some((tok, dn, r)) => {
            println!(
                "\nBEST single token: idx {tok} depth {dn} → correct tool rank {r}. Leaderboard:"
            );
            let didx = match dn {
                "syn" => 0,
                "sem" => 1,
                _ => 2,
            };
            let sid_to_name: HashMap<SectionId, &str> = name_to_sid
                .iter()
                .map(|(n, &id)| (id, n.as_str()))
                .collect();
            let pv = depths[didx].1[tok].as_u128();
            let mut board: Vec<(&str, u32)> = section_sigs
                .iter()
                .filter_map(|(sid, arr)| {
                    let m = arr[didx]
                        .iter()
                        .map(|s| (!(pv ^ s.as_u128())).count_ones())
                        .max()
                        .unwrap_or(0);
                    Some((*sid_to_name.get(sid)?, m))
                })
                .collect();
            board.sort_by(|a, b| b.1.cmp(&a.1));
            for (name, ag) in board.iter().take(8) {
                let mark = if *name == target_tool {
                    "  <-- CORRECT"
                } else {
                    ""
                };
                println!("    {ag:>3}  {name}{mark}");
            }
        }
        None => println!("\nNo masked ranking computed."),
    }

    // Consecutive n-gram match: require L consecutive assistant tokens to align
    // with L consecutive section tokens. A long definition can win a single-token
    // max by chance, but a *run* of aligned tokens (the tool name/phrase) is much
    // harder to hit spuriously — the user's "consecutive match" idea.
    println!(
        "\nconsecutive n-gram match — correct tool's rank by best L-gram agreement (chance≈47):"
    );
    for (dname, dvec, didx) in &depths {
        let pv: Vec<u128> = dvec[ass..nsig].iter().map(|s| s.as_u128()).collect();
        let mut row = String::new();
        for l in 1..=4usize {
            let mut scored: Vec<(SectionId, u32)> = section_sigs
                .iter()
                .map(|(sid, arr)| {
                    let sv: Vec<u128> = arr[*didx].iter().map(|s| s.as_u128()).collect();
                    let mut best = 0u32;
                    if pv.len() >= l && sv.len() >= l {
                        for i in 0..=pv.len() - l {
                            for j in 0..=sv.len() - l {
                                let mut sum = 0u32;
                                for k in 0..l {
                                    sum += (!(pv[i + k] ^ sv[j + k])).count_ones();
                                }
                                best = best.max(sum);
                            }
                        }
                    }
                    (*sid, best)
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1));
            let rank = scored
                .iter()
                .position(|(s, _)| *s == correct_sid)
                .map_or(0, |x| x + 1);
            row.push_str(&format!("  L{l}=#{rank:<2}"));
        }
        println!("  {dname:<4}{row}");
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    // ── 1. Open the substrate model-free.
    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate at {}: {e}", workspace.display()))?;
    eprintln!("opened substrate at {}", workspace.display());

    // ── 2. Build the projection schema + install the catalog, exactly as the
    // daemon does, so the tool SectionIds match the substrate's sections (matched
    // by debug_name = tool name below). ChatML dialect for Qwen3 is derived from
    // the model config without loading any weights.
    let dialect = Model::Qwen3_30B_A3B_Q4
        .builder()
        .conversation_config()
        .dialect;
    let mut proj = Builder::from_yaml_with_vars_and_dialect(
        ZEND_YAML,
        &[("workspace", "candle")],
        Some(&dialect),
    )
    .map_err(|e| anyhow::anyhow!("projection parse: {e}"))?;
    let tok_path = workspace.join(".substrate").join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tok_path.display()))?;
    proj.tokenize_templates::<String, _>(|s| {
        tokenizer
            .encode(s, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| e.to_string())
    })
    .map_err(|e| anyhow::anyhow!("tokenize templates: {e}"))?;
    let dialogue = proj
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("no dialogue layer"))?;
    let group = proj
        .id_for_group("primary_conversation")
        .ok_or_else(|| anyhow::anyhow!("no primary_conversation group"))?;
    let tools = install_tool_catalog(&mut proj, dialogue)?;
    let name_to_sid: HashMap<String, SectionId> =
        tools.iter().map(|(n, id, _)| (n.clone(), *id)).collect();
    eprintln!("installed {} tools into schema", tools.len());

    // ── 3. Rebuild a ProvenanceFile + section corpus from the substrate's
    // tool-section `Signatures` records (matched to tools by debug_name).
    let provenance = ProvenanceFile::new()?;
    let section_streams: Vec<(StreamId, String)> = substrate
        .all_streams()
        .filter_map(|(sid, entry)| match &entry.decl {
            Some(StreamDecl::PromptSection(d)) => Some((sid, d.debug_name.clone())),
            _ => None,
        })
        .collect();

    let mut corpus: Vec<(SectionId, Vec<SigEntry>)> = Vec::new();
    // Per-section raw per-token sigs (syn, sem, prag), kept for the `--diag`
    // per-token agreement analysis that bypasses the aggregate scanner.
    let mut section_sigs: HashMap<SectionId, [Vec<TokenSignature>; 3]> = HashMap::new();
    let bl = TokenSignature::BYTE_LEN;
    for (stream_id, debug_name) in section_streams {
        let Some(&section_id) = name_to_sid.get(&debug_name) else {
            continue; // not a tool section (prelude / summary / etc.)
        };
        let Some(payload) = persistence
            .read_signatures(&substrate, stream_id)
            .map_err(|e| anyhow::anyhow!("read_signatures {debug_name}: {e}"))?
        else {
            continue;
        };
        let mut entries: Vec<SigEntry> = Vec::new();
        for (tc, bytes) in decode_signatures(&payload)
            .map_err(|e| anyhow::anyhow!("decode_signatures {debug_name}: {e}"))?
        {
            let n = tc as usize;
            if bytes.len() < 3 * n * bl {
                continue;
            }
            let parse = |s: &[u8]| -> Vec<TokenSignature> {
                s.chunks_exact(bl)
                    .map(|c| TokenSignature::from_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let syn = parse(&bytes[0..n * bl]);
            let sem = parse(&bytes[n * bl..2 * n * bl]);
            let prag = parse(&bytes[2 * n * bl..3 * n * bl]);
            let acc = section_sigs.entry(section_id).or_default();
            acc[0].extend(syn.iter().copied());
            acc[1].extend(sem.iter().copied());
            acc[2].extend(prag.iter().copied());
            entries.push(provenance.append(&syn, &sem, &prag)?);
        }
        if !entries.is_empty() {
            corpus.push((section_id, entries));
        }
    }
    eprintln!("rebuilt provenance for {} tool sections", corpus.len());
    if corpus.is_empty() {
        anyhow::bail!(
            "no tool-section signatures found in the substrate — is this the daemon workspace?"
        );
    }

    let sid_to_name: HashMap<SectionId, &str> = name_to_sid
        .iter()
        .map(|(n, &id)| (id, n.as_str()))
        .collect();

    // `--diag <tool>`: per-token agreement of one captured assistant turn against
    // every tool section, bypassing the aggregate scanner — to locate which
    // token (and depth) carries the tool signal rather than averaging it away.
    if std::env::args().nth(2).as_deref() == Some("--diag") {
        let tool = std::env::args()
            .nth(3)
            .ok_or_else(|| anyhow::anyhow!("--diag needs a tool name"))?;
        let tc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("candle-conversation")
            .join("tests")
            .join("tool_cases")
            .join("test_config.json");
        let cases: Vec<TestCase> = serde_json::from_str(&std::fs::read_to_string(&tc_path)?)?;
        let prompt_to_tool: HashMap<String, String> =
            cases.into_iter().map(|c| (c.prompt, c.tool)).collect();
        run_diag(
            &substrate,
            &mut persistence,
            &section_sigs,
            &name_to_sid,
            &tool,
            &prompt_to_tool,
        )?;
        return Ok(());
    }

    // `--score2 <L> <window>`: measure consecutive L-gram matching over the
    // assistant name-region window across ALL captured turns. Aggregates top-1 /
    // top-5 / MRR — the real metric for the narrowed+consecutive approach.
    if std::env::args().nth(2).as_deref() == Some("--score2") {
        let l: usize = std::env::args()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let window: usize = std::env::args()
            .nth(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(12);
        let tc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("candle-conversation")
            .join("tests")
            .join("tool_cases")
            .join("test_config.json");
        let cases: Vec<TestCase> = serde_json::from_str(&std::fs::read_to_string(&tc_path)?)?;
        let prompt_to_tool: HashMap<String, String> =
            cases.into_iter().map(|c| (c.prompt, c.tool)).collect();
        score_ngram(
            &substrate,
            &mut persistence,
            &section_sigs,
            &name_to_sid,
            &prompt_to_tool,
            l,
            window,
        )?;
        return Ok(());
    }

    let cfg = Formula::production();

    // ── 4+5. Build scanned probes. Two sources:
    //   `--turns [user|asst]`  → our clean 2-turn captures from the substrate,
    //                            scanning one half of each turn;
    //   else                   → the legacy fixture (single-tool-context prompt
    //                            Q), optionally overridden by a 2nd-arg dir.
    let probes: Vec<Probe> = if std::env::args().nth(2).as_deref() == Some("--turns") {
        let half = std::env::args()
            .nth(3)
            .unwrap_or_else(|| "asst".to_string());
        eprintln!("probes from captured TURNS, half = {half}");
        let tc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("candle-conversation")
            .join("tests")
            .join("tool_cases")
            .join("test_config.json");
        let cases: Vec<TestCase> = serde_json::from_str(&std::fs::read_to_string(&tc_path)?)?;
        let prompt_to_tool: HashMap<String, String> =
            cases.into_iter().map(|c| (c.prompt, c.tool)).collect();
        let probes = build_turn_probes(
            &substrate,
            &mut persistence,
            &provenance,
            &corpus,
            &cfg,
            &half,
            &prompt_to_tool,
            &name_to_sid,
        )?;
        eprintln!("built {} turn probes (half={half})", probes.len());
        probes
    } else {
        let fixture_dir = std::env::args()
            .nth(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("candle-conversation")
                    .join("tests")
                    .join("tool_provenance_real_data")
            });
        eprintln!("probes from {}", fixture_dir.display());
        let manifest: Manifest =
            serde_json::from_str(&std::fs::read_to_string(fixture_dir.join("MANIFEST.json"))?)?;
        let probe_pf = ProvenanceFile::open(fixture_dir.join("signatures.prov"))?;
        let mut probes: Vec<Probe> = Vec::new();
        for s in &manifest.scenarios {
            if s.case_type != "positive" {
                continue;
            }
            let Some(tool) = s.tool.as_deref() else {
                continue;
            };
            if !name_to_sid.contains_key(tool) {
                continue;
            }
            let (syn, sem, prag) = probe_pf.read_entry(SigEntry {
                byte_offset: s.byte_offset,
                token_count: s.token_count,
            })?;
            let mut scanner = BdpScanner::new()
                .with_span_alpha(cfg.span_alpha)
                .with_top_k(cfg.scan_top_k)
                .with_hit_threshold(cfg.hit_threshold);
            scanner.scan_sections(&provenance, &syn, &sem, &prag, &corpus)?;
            let scores: HashMap<SectionId, PerDepthScores> = corpus
                .iter()
                .filter_map(|(sid, _)| Some((*sid, *scanner.section_scores().get(sid)?)))
                .collect();
            probes.push(Probe {
                id: s.id.clone(),
                prompt: s.user_prompt.clone(),
                expected: tool.to_string(),
                scores,
                proj_scores: scanner.to_projection_scores(),
            });
        }
        eprintln!("scanned {} positive prompts", probes.len());
        probes
    };

    // ── 6. Production-formula evaluation with full capture.
    let timeline = TimelineAllocator::default().next();
    let target = ProjectionTarget {
        layer: dialogue,
        group,
        timeline,
    };

    let mut captures: Vec<ProbeCapture> = Vec::new();
    println!(
        "\n=== production formula: prag.span ≥ {:.2}, top-{} ===",
        cfg.score_threshold, cfg.select_k
    );
    println!(
        "{:<22} {:<12} {:>5} {:>9}  {:<7} selected",
        "prompt-id", "expected", "rank", "score", "thr?"
    );
    for p in &probes {
        let ranked = rank_tools(&p.scores, &sid_to_name, cfg.formula, &cfg.weights);
        let expected_rank = ranked
            .iter()
            .position(|(n, _)| *n == p.expected)
            .map(|i| i + 1)
            .unwrap_or(ranked.len() + 1);
        let expected_score = ranked
            .iter()
            .find(|(n, _)| *n == p.expected)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        // Manual selection: score ≥ threshold, then top select_k (already sorted).
        let selected: Vec<String> = ranked
            .iter()
            .filter(|(_, s)| *s >= cfg.score_threshold)
            .take(cfg.select_k)
            .map(|(n, _)| n.to_string())
            .collect();
        let correct = selected.iter().any(|n| n == &p.expected);
        let passed_threshold = expected_score >= cfg.score_threshold;

        // Cross-check against the real Decode projection.
        let scored = ScoredSubstrate::new(&substrate, &p.proj_scores);
        let projection_selected: Vec<String> = proj
            .project_with_mode(target, &scored, ProjectionMode::Decode)
            .sealed_sections()
            .filter_map(|rs| sid_to_name.get(&rs.id).map(|s| s.to_string()))
            .collect();

        let exp_pds = p.scores.get(name_to_sid.get(&p.expected).unwrap());
        let (top_distractor, top_distractor_breakdown) = ranked
            .iter()
            .find(|(n, _)| *n != p.expected)
            .map(|(n, s)| {
                let pds = p.scores.get(name_to_sid.get(*n).unwrap());
                (
                    Some(Ranked {
                        tool: n.to_string(),
                        score: *s,
                    }),
                    pds.map(Depths::of),
                )
            })
            .unwrap_or((None, None));

        println!(
            "{:<22} {:<12} {:>5} {:>9.1}  {:<7} {:?}",
            p.id,
            p.expected,
            expected_rank,
            expected_score,
            if passed_threshold { "yes" } else { "no" },
            selected,
        );

        captures.push(ProbeCapture {
            id: p.id.clone(),
            prompt: p.prompt.clone(),
            expected: p.expected.clone(),
            expected_rank,
            expected_score,
            passed_threshold,
            selected,
            correct,
            projection_selected,
            top: ranked
                .iter()
                .take(10)
                .map(|(n, s)| Ranked {
                    tool: n.to_string(),
                    score: *s,
                })
                .collect(),
            expected_breakdown: exp_pds
                .map(Depths::of)
                .unwrap_or_else(|| Depths::of(&PerDepthScores::default())),
            top_distractor,
            top_distractor_breakdown,
        });
    }

    // Aggregate summary for the production formula.
    let n = captures.len().max(1);
    let summary = Summary {
        n: captures.len(),
        top1: captures.iter().filter(|c| c.expected_rank == 1).count() as f32 / n as f32,
        top_k: captures.iter().filter(|c| c.correct).count() as f32 / n as f32,
        mrr: captures
            .iter()
            .map(|c| 1.0 / c.expected_rank as f32)
            .sum::<f32>()
            / n as f32,
        threshold_pass: captures.iter().filter(|c| c.passed_threshold).count() as f32 / n as f32,
        mean_rank: captures.iter().map(|c| c.expected_rank as f32).sum::<f32>() / n as f32,
    };
    println!(
        "\nproduction: Top-1 {:.0}%  Top-{} {:.0}%  MRR {:.3}  threshold-pass {:.0}%  mean-rank {:.1}",
        summary.top1 * 100.0,
        cfg.select_k,
        summary.top_k * 100.0,
        summary.mrr,
        summary.threshold_pass * 100.0,
        summary.mean_rank,
    );

    // Write the full debug capture.
    let report = Report {
        config: ConfigDump {
            span_alpha: cfg.span_alpha,
            scan_top_k: cfg.scan_top_k,
            hit_threshold: cfg.hit_threshold,
            formula: format!("{:?}", cfg.formula),
            weights: [
                cfg.weights.syntactic,
                cfg.weights.semantic,
                cfg.weights.pragmatic,
            ],
            score_threshold: cfg.score_threshold,
            select_k: cfg.select_k,
        },
        summary,
        probes: captures,
    };
    let out_path = std::env::current_dir()?.join("tool_select_debug.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
    println!("full per-prompt capture → {}", out_path.display());

    // ── 7. Formula sweep: which (statistic × depth-weight) best separates the
    // right tool? Pure ranking accuracy (threshold-independent), over the cached
    // scans. This is the optimisation signal.
    let stats: [(&str, ScoreFormula); 7] = [
        ("max", ScoreFormula::Max),
        ("sum", ScoreFormula::Sum),
        ("mean", ScoreFormula::Mean),
        ("top_k_mean", ScoreFormula::TopKMean { k: 8 }),
        ("count", ScoreFormula::Count),
        ("span", ScoreFormula::Span { alpha: 2.0 }),
        ("pertok_excess", ScoreFormula::PerTokenExcess),
    ];
    let w = |s: f32, m: f32, p: f32| DepthWeights {
        syntactic: s,
        semantic: m,
        pragmatic: p,
    };
    let wsets: [(&str, DepthWeights); 6] = [
        ("prag", w(0.0, 0.0, 1.0)),
        ("1/1/4", w(1.0, 1.0, 4.0)),
        ("syn", w(1.0, 0.0, 0.0)),
        ("sem", w(0.0, 1.0, 0.0)),
        ("sem+prag", w(0.0, 3.0, 4.0)),
        ("1/1/1", w(1.0, 1.0, 1.0)),
    ];

    println!(
        "\n=== formula sweep (ranking accuracy over {} prompts) ===",
        probes.len()
    );
    println!(
        "{:<14} {:<10} {:>7} {:>7} {:>7}",
        "statistic", "weights", "Top-1", "Top-5", "MRR"
    );
    let mut rows: Vec<(String, f32, f32, f32)> = Vec::new();
    for (sname, f) in stats {
        for (wname, ws) in &wsets {
            let mut top1 = 0usize;
            let mut top5 = 0usize;
            let mut mrr = 0.0f32;
            for p in &probes {
                let ranked = rank_tools(&p.scores, &sid_to_name, f, ws);
                let rank = ranked
                    .iter()
                    .position(|(n, _)| *n == p.expected)
                    .map(|i| i + 1)
                    .unwrap_or(ranked.len() + 1);
                if rank == 1 {
                    top1 += 1;
                }
                if rank <= 5 {
                    top5 += 1;
                }
                mrr += 1.0 / rank as f32;
            }
            let np = probes.len().max(1) as f32;
            rows.push((
                format!("{sname:<14} {wname:<10}"),
                top1 as f32 / np,
                top5 as f32 / np,
                mrr / np,
            ));
        }
    }
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (label, top1, top5, mrr) in &rows {
        println!(
            "{label} {:>6.0}% {:>6.0}% {:>7.3}",
            top1 * 100.0,
            top5 * 100.0,
            mrr
        );
    }

    Ok(())
}
