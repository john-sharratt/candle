//! Focused, fast provenance-retrieval harness — §77 (leave-one-conversation-out) and
//! §78 (k-fold, probe/gallery disjoint by conversation) over the per-projection rolling
//! wide-Q windows.
//!
//! Extracted from `calibrate_alignment` so it (a) compiles in seconds instead of ~48s,
//! (b) skips the §16–21 CCA reconstruction preamble entirely, and (c) scans a single
//! contiguous `u64` buffer (cache-friendly popcount) instead of `Vec<Vec<u64>>`. Every
//! phase is timed so it's obvious where any remaining cost is.
//!
//! ```text
//! cargo run -p zend --example provenance_probe --release -- [workspace]
//!   BACK=64      rolling lookback tokens (default 64)
//!   S78_KS=4,3,2 k-fold values to sweep (default 4)
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::{decode_events, SystemItem};
use candle_conversation::provenance::{
    decode_wide_sigs, score_provenance_late_fusion, GroupBudget, SectionPolicy, SectionSelector,
    WideQSig,
};
use candle_conversation::substrate::Substrate;
use rayon::prelude::*;

/// KV heads per layer in the wide-Q signature (`n_kv_head`).
const HEADS_PER_LAYER: usize = 4;
/// Layers in the model / wide-Q stack.
const N_LAYERS: usize = 48;

/// Keep only layers `[lo, hi)` of a token's words (each layer = `words.len()/N_LAYERS`
/// u64). Used to test which layers actually carry the retrieval signal.
fn select_layers(words: &[u64], lo: usize, hi: usize) -> Vec<u64> {
    let wpl = words.len() / N_LAYERS;
    let (a, b) = (lo.min(N_LAYERS) * wpl, hi.min(N_LAYERS) * wpl);
    words[a.min(words.len())..b.min(words.len())].to_vec()
}

/// Rotate a head's 128 sign bits (`w0` = dims 0–63, `w1` = dims 64–127) left by `r`.
fn rol128(w0: u64, w1: u64, r: usize) -> (u64, u64) {
    let r = (r % 128) as u32;
    if r == 0 {
        return (w0, w1);
    }
    let v = (w0 as u128) | ((w1 as u128) << 64);
    let rot = v.rotate_left(r);
    (rot as u64, (rot >> 64) as u64)
}

/// Collapse the layer stack by XORing consecutive groups of `group` layers together,
/// preserving the per-(head, dim) bit structure. `words` is one token's
/// `[layer][head][wph]`-ordered sign bits. Each layer at position `p` within its group
/// is rotated left by `p*shift` bits inside its 128-bit head before the XOR — so with
/// `shift > 0` the layers are staggered into different phases instead of colliding
/// dim-aligned. `shift = 0` is the plain dim-aligned fold. Returns
/// `ceil(n_layers/group)` layer-groups' worth of words. `group <= 1` returns a copy.
fn fold_layers(words: &[u64], n_heads: usize, group: usize, shift: usize) -> Vec<u64> {
    if group <= 1 || n_heads == 0 {
        return words.to_vec();
    }
    let wph = words.len() / n_heads;
    let words_per_layer = HEADS_PER_LAYER * wph;
    let n_layers = n_heads / HEADS_PER_LAYER;
    let n_groups = n_layers.div_ceil(group);
    let mut out = vec![0u64; n_groups * words_per_layer];
    for g in 0..n_groups {
        for (p, l) in ((g * group)..((g + 1) * group).min(n_layers)).enumerate() {
            let r = p * shift;
            for h in 0..HEADS_PER_LAYER {
                let bi = (l * HEADS_PER_LAYER + h) * wph;
                let bo = g * words_per_layer + h * wph;
                if wph == 2 && shift > 0 {
                    let (w0, w1) = rol128(words[bi], words[bi + 1], r);
                    out[bo] ^= w0;
                    out[bo + 1] ^= w1;
                } else {
                    for i in 0..wph {
                        out[bo + i] ^= words[bi + i];
                    }
                }
            }
        }
    }
    out
}

/// Fold layers into VARIABLE-size groups given by `sizes` (bottom→top): group `g` XORs
/// its `sizes[g]` layers, each rotated by `position × shift` (the decorrelating
/// stagger). Small groups = high resolution where the signal is; big groups = heavy
/// compression where it's weak. Produces one 512-bit (4-head) block per group.
fn fold_dist(
    words: &[u64],
    n_heads: usize,
    sizes: &[usize],
    shift: usize,
    headfold: bool,
    head_shift: usize,
) -> Vec<u64> {
    let wph = words.len() / n_heads;
    let n_layers = n_heads / HEADS_PER_LAYER;
    // With head-fold, a group collapses to ONE 128-bit head-aggregate (wph words);
    // otherwise it keeps all 4 heads (4·wph words).
    let words_per_group = if headfold { wph } else { HEADS_PER_LAYER * wph };
    let mut out = vec![0u64; sizes.len() * words_per_group];
    let mut l0 = 0usize;
    for (g, &sz) in sizes.iter().enumerate() {
        for (p, l) in (l0..(l0 + sz).min(n_layers)).enumerate() {
            for h in 0..HEADS_PER_LAYER {
                let bi = (l * HEADS_PER_LAYER + h) * wph;
                if headfold {
                    // XOR all 4 heads (staggered by h·head_shift) AND all layers
                    // (staggered by p·shift) into one 128-bit group aggregate.
                    let bo = g * words_per_group;
                    if wph == 2 {
                        let (w0, w1) = rol128(words[bi], words[bi + 1], p * shift + h * head_shift);
                        out[bo] ^= w0;
                        out[bo + 1] ^= w1;
                    } else {
                        for i in 0..wph {
                            out[bo + i] ^= words[bi + i];
                        }
                    }
                } else {
                    let bo = g * words_per_group + h * wph;
                    if wph == 2 && shift > 0 {
                        let (w0, w1) = rol128(words[bi], words[bi + 1], p * shift);
                        out[bo] ^= w0;
                        out[bo + 1] ^= w1;
                    } else {
                        for i in 0..wph {
                            out[bo + i] ^= words[bi + i];
                        }
                    }
                }
            }
        }
        l0 += sz;
        if l0 >= n_layers {
            break;
        }
    }
    out
}

/// Named group-size distributions (bottom→top), each summing to 48 layers.
fn dist_set() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("uniform3", vec![3; 16]),
        ("uniform4", vec![4; 12]),
        ("uniform6", vec![6; 8]),
        ("uniform8", vec![8; 6]),
        ("uniform16", vec![16; 3]),
        ("edges", vec![1, 2, 20, 2, 20, 2, 1]),
        ("pyramid", vec![1, 2, 4, 8, 18, 8, 4, 2, 1]),
        ("topfine", vec![24, 12, 6, 3, 2, 1]),
        ("topfine2", vec![16, 10, 8, 6, 4, 2, 1, 1]),
        ("botfine", vec![1, 2, 3, 6, 12, 24]),
        ("midfine", vec![12, 6, 3, 2, 1, 1, 2, 3, 6, 12]),
    ]
}

/// Growing little-endian bit buffer for packing variable-width per-head signatures.
struct BitBuf {
    words: Vec<u64>,
    nbits: usize,
}
impl BitBuf {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            nbits: 0,
        }
    }
    /// Append the low `n` bits (n ≤ 64) of `val`.
    fn push(&mut self, val: u64, n: usize) {
        if n == 0 {
            return;
        }
        let val = if n >= 64 {
            val
        } else {
            val & ((1u64 << n) - 1)
        };
        let w = self.nbits / 64;
        let off = self.nbits % 64;
        while self.words.len() <= w + 1 {
            self.words.push(0);
        }
        self.words[w] |= val << off;
        if off + n > 64 {
            self.words[w + 1] |= val >> (64 - off);
        }
        self.nbits += n;
    }
    fn into_words(self) -> Vec<u64> {
        self.words
    }
}

/// Block-fold a head's 128 sign bits (`w0` = dims 0–63, `w1` = dims 64–127) down to
/// `b` bits (`b ∈ {1,2,4,8,16,32,64}`): output bit `i` = XOR of the i-th contiguous
/// block of `128/b` dims. This is the intra-head XOR compression.
fn fold128(w0: u64, w1: u64, b: usize) -> u64 {
    let s = 128 / b;
    let mut out = 0u64;
    for i in 0..b {
        let mut acc = 0u64;
        for j in 0..s {
            let p = i * s + j;
            let bit = if p < 64 {
                (w0 >> p) & 1
            } else {
                (w1 >> (p - 64)) & 1
            };
            acc ^= bit;
        }
        out |= acc << i;
    }
    out
}

/// Asymmetric per-token signature from a per-layer bit budget: each layer's 4 heads
/// are each folded from 128 bits down to `budget[layer]` bits (0 = drop the layer,
/// 128 = full resolution), concatenated. Spends Hamming-comparison bits where the
/// signal is — full resolution on informative (top) layers, heavy XOR compression on
/// weak (bottom) layers.
fn build_sig(tok_words: &[u64], budget: &[usize]) -> Vec<u64> {
    let mut buf = BitBuf::new();
    for (layer, &b) in budget.iter().enumerate() {
        if b == 0 {
            continue;
        }
        for head in 0..HEADS_PER_LAYER {
            let base = (layer * HEADS_PER_LAYER + head) * 2;
            if base + 1 >= tok_words.len() {
                break;
            }
            let (w0, w1) = (tok_words[base], tok_words[base + 1]);
            if b >= 128 {
                buf.push(w0, 64);
                buf.push(w1, 64);
            } else {
                buf.push(fold128(w0, w1, b), b);
            }
        }
    }
    buf.into_words()
}

/// Named bit-budget schedules over the 48 layers (index 0 = bottom … 47 = top).
/// `(lo, hi, bits)` sets layers `[lo, hi)` to `bits` bits/head; unset layers = 0.
fn scheme_budgets() -> Vec<(&'static str, [usize; 48])> {
    let seg = |ranges: &[(usize, usize, usize)]| -> [usize; 48] {
        let mut b = [0usize; 48];
        for &(lo, hi, bits) in ranges {
            for l in lo..hi.min(N_LAYERS) {
                b[l] = bits;
            }
        }
        b
    };
    vec![
        ("full", seg(&[(0, 48, 128)])),
        ("top8", seg(&[(40, 48, 128)])),
        ("top4", seg(&[(44, 48, 128)])),
        ("top2", seg(&[(46, 48, 128)])),
        ("fold32", seg(&[(0, 40, 32), (40, 48, 128)])),
        ("topheavy16", seg(&[(0, 40, 16), (40, 48, 128)])),
        ("step", seg(&[(0, 16, 8), (16, 32, 32), (32, 48, 128)])),
        (
            "quart",
            seg(&[(0, 12, 8), (12, 24, 16), (24, 36, 32), (36, 48, 128)]),
        ),
        (
            "geo",
            seg(&[(0, 24, 8), (24, 36, 32), (36, 44, 64), (44, 48, 128)]),
        ),
        (
            "invramp",
            seg(&[(0, 12, 128), (12, 24, 32), (24, 36, 16), (36, 48, 8)]),
        ),
    ]
}

/// Family stem: `telnet_session_list` → `session_list`.
fn stem(n: &str) -> String {
    let p: Vec<&str> = n.rsplitn(3, '_').collect();
    if p.len() >= 2 {
        format!("{}_{}", p[1], p[0])
    } else {
        n.to_string()
    }
}

/// One retrieval probe: the rolling wide-Q window ending at a projection point, the
/// `tools` section that projection locked, and the conversation it fired in.
struct ProbeCase {
    tool: String,
    conv: u64,
    /// Projection point in the turn's token stream (`assistant_content_start + end_token`).
    /// Orders a conversation's projections in decode order for the online walk.
    point: usize,
    window: Vec<WideQSig>,
}

/// Build per-projection probe cases: for every projection event that locked a `tools`
/// section, the lookup signature is the last `back` wide-Q tokens ending at the
/// projection point (`assistant_content_start + end_token`).
fn projection_probe_cases(substrate: &Substrate, back: usize) -> Vec<ProbeCase> {
    let mut cases = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(d)) = &e.decl else {
            continue;
        };
        let Some(history) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if history.is_empty() {
            continue;
        }
        let Some(events) = e.projection_events.as_ref().map(|b| decode_events(b)) else {
            continue;
        };
        let asst = candle_conversation::turn_layout::TurnLayout::new(d.segments.clone())
            .assistant_content_start() as usize;
        for ev in &events {
            let Some(tool) = ev.selection.system.iter().find_map(|item| match item {
                SystemItem::Collection { name, sections } if name == "tools" => {
                    sections.iter().find(|s| s.selected).map(|s| s.name.clone())
                }
                _ => None,
            }) else {
                continue;
            };
            // Point-model event: `start_token` is the generated position at which
            // this projection was selected (the old span model's `end_token`).
            let point = (asst + ev.start_token as usize).min(history.len());
            let lo = point.saturating_sub(back);
            let window = history[lo..point].to_vec();
            if window.is_empty() {
                continue;
            }
            cases.push(ProbeCase {
                tool,
                conv: sid.0,
                point,
                window,
            });
        }
    }
    cases
}

/// Stable tool index over the case set. Returns `(n_tools, case→tool_id, stem_per_id)`.
fn build_tool_index(cases: &[ProbeCase]) -> (usize, Vec<usize>, Vec<String>) {
    let mut ids: HashMap<&str, usize> = HashMap::new();
    for c in cases {
        let n = ids.len();
        ids.entry(c.tool.as_str()).or_insert(n);
    }
    let n_tools = ids.len();
    let case_tool: Vec<usize> = cases.iter().map(|c| ids[c.tool.as_str()]).collect();
    let mut stem_id = vec![String::new(); n_tools];
    for c in cases {
        stem_id[ids[c.tool.as_str()]] = stem(&c.tool).to_string();
    }
    (n_tools, case_tool, stem_id)
}

/// For every projection (case), scan its probe window against the full gallery of all
/// OTHER conversations' tokens (leave-one-conversation-out) and accumulate a per-tool
/// confidence (sum-of-z: each group's best gallery-token match credits its tool). The
/// returned vector is `n_cases × n_tools`.
fn precompute_proj_scores(
    cases: &[ProbeCase],
    case_conv: &[u64],
    case_tool: &[usize],
    n_tools: usize,
) -> Vec<Vec<f32>> {
    let w = cases[0].window[0].words.len();
    let n_groups = cases[0].window[0].n_heads as usize / 4;
    let gw = 8usize;
    let total_tok: usize = cases.iter().map(|c| c.window.len()).sum();
    let mut gwords: Vec<u64> = Vec::with_capacity(total_tok * w);
    let mut gconv: Vec<u64> = Vec::with_capacity(total_tok);
    let mut gtool: Vec<usize> = Vec::with_capacity(total_tok);
    for (ci, c) in cases.iter().enumerate() {
        for tok in &c.window {
            gwords.extend_from_slice(&tok.words);
            gconv.push(case_conv[ci]);
            gtool.push(case_tool[ci]);
        }
    }
    let n_gal = gconv.len();
    (0..cases.len())
        .into_par_iter()
        .map(|a| {
            let qconv = case_conv[a];
            let mut ts = vec![0f32; n_tools];
            for q in &cases[a].window {
                for g in 0..n_groups {
                    let base = g * gw;
                    let qg = &q.words[base..base + gw];
                    let (mut best, mut btool) = (0u32, usize::MAX);
                    let (mut sum, mut sumsq, mut cnt) = (0u64, 0u64, 0u64);
                    for j in 0..n_gal {
                        if gconv[j] == qconv {
                            continue;
                        }
                        let cw = &gwords[j * w + base..j * w + base + gw];
                        let mut ag = 0u32;
                        for kk in 0..8 {
                            ag += (!(qg[kk] ^ cw[kk])).count_ones();
                        }
                        if ag > best {
                            best = ag;
                            btool = gtool[j];
                        }
                        sum += ag as u64;
                        sumsq += (ag as u64) * (ag as u64);
                        cnt += 1;
                    }
                    if btool != usize::MAX {
                        let nn = cnt as f32;
                        let mean = sum as f32 / nn;
                        let var = (sumsq as f32 / nn - mean * mean).max(1e-6);
                        let z = ((best as f32 - mean) / var.sqrt()).max(0.0);
                        ts[btool] += z;
                    }
                }
            }
            ts
        })
        .collect()
}

/// A belief-update rule applied per projection step over a walk. Each merges the fresh
/// per-tool score into a running accumulator; the argmax at the end is the prediction.
#[derive(Clone, Copy)]
enum Mech {
    /// Pool everything, no decay (`acc += s`). Best for single-intent, blind to switches.
    Sum,
    /// Keep the strongest score each tool ever hit (`acc = max(acc, s)`).
    Max,
    /// Classic EWMA leak (`acc = λ·acc + s`) — decays the whole belief each step.
    Mult(f32),
    /// Decay the running max but let a fresh score overwrite it (`acc = max(λ·acc, s)`).
    LeakyMax(f32),
    /// Additive-with-delay: subtract a share of the current leader then add fresh
    /// (`acc = max(0, acc − β·max) + s`). Suppresses stale followers, keeps the leader.
    RelLeak(f32),
    /// Additive, later projections weighted more (`acc += (1+g·step)·s`). Never forgets,
    /// just tilts toward recency.
    Ramp(f32),
    /// EWMA leak then renormalise to a simplex (`acc = λ·acc + s; acc /= Σacc`).
    Simplex(f32),
    /// Two timescales: a fast leaky belief plus an un-decayed slow pool, summed at the
    /// end with weight α (`fast = λ·fast + s; slow += s; pred = fast + α·slow`).
    TwoScale(f32, f32),
    /// Surprise-gated: pool (`acc += s`) while the fresh leader agrees with the running
    /// leader; decay by `d` first only when the fresh top tool DISAGREES. Zero decay
    /// within a stable topic, hard forget on a switch.
    Surprise(f32),
    /// Surprise-gated max-merge: `acc = max(acc, s)` on agreement, `acc = max(d·acc, s)`
    /// on disagreement.
    SurpMax(f32),
}

/// Argmax of `v` if it carries any positive mass, else `None`.
fn argmax_mass(v: &[f32]) -> Option<usize> {
    let mut bi = usize::MAX;
    let mut bv = 0f32;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi != usize::MAX).then_some(bi)
}

/// Walk `projs` (indices into `proj_score`) under `mech`, returning the final per-tool
/// accumulator to argmax over.
fn walk_final(projs: &[usize], proj_score: &[Vec<f32>], n_tools: usize, mech: &Mech) -> Vec<f32> {
    let mut acc = vec![0f32; n_tools];
    let mut slow = vec![0f32; n_tools];
    for (step, &pj) in projs.iter().enumerate() {
        let s = &proj_score[pj];
        match *mech {
            Mech::Sum => {
                for t in 0..n_tools {
                    acc[t] += s[t];
                }
            }
            Mech::Max => {
                for t in 0..n_tools {
                    acc[t] = acc[t].max(s[t]);
                }
            }
            Mech::Mult(l) => {
                for t in 0..n_tools {
                    acc[t] = l * acc[t] + s[t];
                }
            }
            Mech::LeakyMax(l) => {
                for t in 0..n_tools {
                    acc[t] = (l * acc[t]).max(s[t]);
                }
            }
            Mech::RelLeak(b) => {
                let m = acc.iter().copied().fold(0f32, f32::max);
                for t in 0..n_tools {
                    acc[t] = (acc[t] - b * m).max(0.0) + s[t];
                }
            }
            Mech::Ramp(g) => {
                let wt = 1.0 + g * step as f32;
                for t in 0..n_tools {
                    acc[t] += wt * s[t];
                }
            }
            Mech::Simplex(l) => {
                for t in 0..n_tools {
                    acc[t] = l * acc[t] + s[t];
                }
                let sm: f32 = acc.iter().sum();
                if sm > 0.0 {
                    for t in 0..n_tools {
                        acc[t] /= sm;
                    }
                }
            }
            Mech::TwoScale(lf, _) => {
                for t in 0..n_tools {
                    acc[t] = lf * acc[t] + s[t];
                    slow[t] += s[t];
                }
            }
            Mech::Surprise(d) => {
                let jt = argmax_mass(s);
                if let (Some(c), Some(j)) = (argmax_mass(&acc), jt) {
                    if j != c {
                        for x in acc.iter_mut() {
                            *x *= d;
                        }
                    }
                }
                for t in 0..n_tools {
                    acc[t] += s[t];
                }
            }
            Mech::SurpMax(d) => {
                let jt = argmax_mass(s);
                if let (Some(c), Some(j)) = (argmax_mass(&acc), jt) {
                    if j != c {
                        for x in acc.iter_mut() {
                            *x *= d;
                        }
                    }
                }
                for t in 0..n_tools {
                    acc[t] = acc[t].max(s[t]);
                }
            }
        }
    }
    if let Mech::TwoScale(_, a) = *mech {
        for t in 0..n_tools {
            acc[t] += a * slow[t];
        }
    }
    acc
}

/// Flattened corpus: contiguous `words` (`n_tok × w`), plus per-token / per-case keys.
struct Flat {
    case_tool: Vec<String>,
    case_conv: Vec<u64>,
    case_gconv: Vec<u32>,            // per-case conversation index (0..n_convs)
    words: Vec<u64>,                 // n_tok * w
    tok_case: Vec<u32>,              // n_tok
    tok_gconv: Vec<u32>,             // n_tok conversation index
    case_range: Vec<(usize, usize)>, // (start, len) of each case's tokens in `words`
    w: usize,
    n_cases: usize,
    n_tok: usize,
}

impl Flat {
    /// Build the flattened corpus, applying `sig_fn` to each token's raw words to
    /// produce its (possibly reduced) signature. All signatures must be the same
    /// length.
    fn new(cases: &[ProbeCase], sig_fn: impl Fn(&[u64]) -> Vec<u64>) -> Flat {
        let case_tool: Vec<String> = cases.iter().map(|c| c.tool.clone()).collect();
        let case_conv: Vec<u64> = cases.iter().map(|c| c.conv).collect();
        // Stable conversation → index map.
        let mut conv_idx: HashMap<u64, u32> = HashMap::new();
        for &cv in &case_conv {
            let n = conv_idx.len() as u32;
            conv_idx.entry(cv).or_insert(n);
        }
        let case_gconv: Vec<u32> = case_conv.iter().map(|cv| conv_idx[cv]).collect();
        // Raw words/token — used to filter malformed tokens.
        let raw_w = cases
            .iter()
            .flat_map(|c| c.window.first())
            .map(|t| t.words.len())
            .next()
            .unwrap_or(0);
        let mut words = Vec::new();
        let mut tok_case = Vec::new();
        let mut tok_gconv = Vec::new();
        let mut case_range = Vec::with_capacity(cases.len());
        for (ci, c) in cases.iter().enumerate() {
            let start = tok_case.len();
            for tok in &c.window {
                if tok.words.len() != raw_w {
                    continue;
                }
                words.extend_from_slice(&sig_fn(&tok.words));
                tok_case.push(ci as u32);
                tok_gconv.push(case_gconv[ci]);
            }
            case_range.push((start, tok_case.len() - start));
        }
        let n_tok = tok_case.len();
        let w = if n_tok > 0 { words.len() / n_tok } else { 0 };
        Flat {
            n_cases: case_tool.len(),
            case_tool,
            case_conv,
            case_gconv,
            words,
            tok_case,
            tok_gconv,
            case_range,
            w,
            n_tok,
        }
    }

    /// Symmetric layer-fold (+ optional layer subset, + per-layer bit `shift`) — the
    /// `LFOLD`/`LSEL`/`SHIFT` path.
    fn build(
        cases: &[ProbeCase],
        group: usize,
        lsel: Option<(usize, usize)>,
        shift: usize,
    ) -> Flat {
        Flat::new(cases, |w| {
            let (sel, n_heads) = match lsel {
                Some((lo, hi)) => (
                    select_layers(w, lo, hi),
                    hi.min(N_LAYERS).saturating_sub(lo) * HEADS_PER_LAYER,
                ),
                None => (w.to_vec(), N_LAYERS * HEADS_PER_LAYER),
            };
            fold_layers(&sel, n_heads, group, shift)
        })
    }
}

/// Per-token consensus vote for case `a`, blocked for cache reuse: read each gallery
/// token ONCE and compare it against all of `a`'s query tokens (which fit in L2), so
/// the 395 MB gallery is streamed once per probe instead of once per query token.
/// `tok_key`/`case_key` are per-token/per-case group ids; a gallery token or a
/// candidate case is excluded iff its key equals `exclude` (the leave-out group).
/// Returns `[Tool-1, Tool-5, Fam-1, Fam-5]` or `None` if unscorable.
fn score_case(
    f: &Flat,
    a: usize,
    tok_key: &[u32],
    case_key: &[u32],
    exclude: u32,
) -> Option<[bool; 4]> {
    let (a_start, a_len) = f.case_range[a];
    if a_len == 0 {
        return None;
    }
    let tool_a = &f.case_tool[a];
    // Scorable only if a same-tool sibling exists in the gallery.
    if !(0..f.n_cases).any(|b| case_key[b] != exclude && &f.case_tool[b] == tool_a) {
        return None;
    }
    let w = f.w;
    let qbuf = &f.words[a_start * w..(a_start + a_len) * w]; // hot in cache
    let mut best_ag = vec![0u32; a_len];
    let mut best_case = vec![u32::MAX; a_len];
    for j in 0..f.n_tok {
        if tok_key[j] == exclude {
            continue;
        }
        let wb = &f.words[j * w..j * w + w];
        let cj = f.tok_case[j];
        for qi in 0..a_len {
            let wa = &qbuf[qi * w..qi * w + w];
            let mut ag = 0u32;
            for k in 0..w {
                ag += (!(wa[k] ^ wb[k])).count_ones();
            }
            if ag > best_ag[qi] {
                best_ag[qi] = ag;
                best_case[qi] = cj;
            }
        }
    }
    let mut votes = vec![0u32; f.n_cases];
    for &bc in &best_case {
        if bc != u32::MAX {
            votes[bc as usize] += 1;
        }
    }
    let mut order: Vec<usize> = (0..f.n_cases).filter(|&b| case_key[b] != exclude).collect();
    order.sort_by(|&x, &y| votes[y].cmp(&votes[x]));
    let st = |b: usize| &f.case_tool[b] == tool_a;
    let sf = |b: usize| stem(&f.case_tool[b]) == stem(tool_a);
    Some([
        st(order[0]),
        order.iter().take(5).any(|&b| st(b)),
        sf(order[0]),
        order.iter().take(5).any(|&b| sf(b)),
    ])
}

/// Weighted-vote variant of [`score_case`]: the query↔gallery agreement is a WEIGHTED
/// sum of per-group sign agreements — `Σ_g weights[g] × popcount(XNOR of group g)` —
/// where each group is `gw` u64 words. So instead of concatenating all groups into one
/// flat popcount (which averages them), the more discriminative groups dominate the
/// argmax. Same blocked, cache-friendly structure as `score_case`.
fn score_case_weighted(
    f: &Flat,
    a: usize,
    tok_key: &[u32],
    case_key: &[u32],
    exclude: u32,
    weights: &[f32],
    gw: usize,
) -> Option<[bool; 4]> {
    let (a_start, a_len) = f.case_range[a];
    if a_len == 0 {
        return None;
    }
    let tool_a = &f.case_tool[a];
    if !(0..f.n_cases).any(|b| case_key[b] != exclude && &f.case_tool[b] == tool_a) {
        return None;
    }
    let w = f.w;
    let n_groups = (w / gw.max(1)).min(weights.len());
    // Only groups with a nonzero weight cost anything — a dropped group is skipped
    // entirely (so solos / sparse weightings scan far fewer words).
    let active: Vec<(usize, f32)> = (0..n_groups)
        .filter(|&g| weights[g] != 0.0)
        .map(|g| (g, weights[g]))
        .collect();
    let qbuf = &f.words[a_start * w..(a_start + a_len) * w];
    let mut best_score = vec![f32::MIN; a_len];
    let mut best_case = vec![u32::MAX; a_len];
    for j in 0..f.n_tok {
        if tok_key[j] == exclude {
            continue;
        }
        let wb = &f.words[j * w..j * w + w];
        let cj = f.tok_case[j];
        for qi in 0..a_len {
            let wa = &qbuf[qi * w..qi * w + w];
            let mut score = 0f32;
            for &(g, wt) in &active {
                let mut ag = 0u32;
                for k in g * gw..(g + 1) * gw {
                    ag += (!(wa[k] ^ wb[k])).count_ones();
                }
                score += wt * ag as f32;
            }
            if score > best_score[qi] {
                best_score[qi] = score;
                best_case[qi] = cj;
            }
        }
    }
    let mut votes = vec![0u32; f.n_cases];
    for &bc in &best_case {
        if bc != u32::MAX {
            votes[bc as usize] += 1;
        }
    }
    let mut order: Vec<usize> = (0..f.n_cases).filter(|&b| case_key[b] != exclude).collect();
    order.sort_by(|&x, &y| votes[y].cmp(&votes[x]));
    let st = |b: usize| &f.case_tool[b] == tool_a;
    let sf = |b: usize| stem(&f.case_tool[b]) == stem(tool_a);
    Some([
        st(order[0]),
        order.iter().take(5).any(|&b| st(b)),
        sf(order[0]),
        order.iter().take(5).any(|&b| sf(b)),
    ])
}

/// LATE-fusion variant: instead of combining group agreements into one score and
/// taking a single argmax, **each group votes independently**. For every query token,
/// each active group picks *its own* best-matching gallery token (argmax on that
/// group's bits alone) and adds `weight[g]` to that token's case. Votes are tallied
/// across all query tokens AND all groups — so when several groups point at the same
/// case, it reinforces, letting the combination exceed the best single group.
fn score_case_late(
    f: &Flat,
    a: usize,
    tok_key: &[u32],
    case_key: &[u32],
    exclude: u32,
    weights: &[f32],
    gw: usize,
    conf: u8,
    freq: &[f32],
) -> Option<[bool; 4]> {
    let (a_start, a_len) = f.case_range[a];
    if a_len == 0 {
        return None;
    }
    let tool_a = &f.case_tool[a];
    if !(0..f.n_cases).any(|b| case_key[b] != exclude && &f.case_tool[b] == tool_a) {
        return None;
    }
    let w = f.w;
    let n_groups = (w / gw.max(1)).min(weights.len());
    let active: Vec<(usize, f32)> = (0..n_groups)
        .filter(|&g| weights[g] != 0.0)
        .map(|g| (g, weights[g]))
        .collect();
    let _ = freq;
    let na = active.len();
    // Per active group: its starting word offset (avoids re-deriving in the hot loop).
    let starts: Vec<usize> = active.iter().map(|&(g, _)| g * gw).collect();
    let qbuf = &f.words[a_start * w..(a_start + a_len) * w];
    // Hot loop: const-8 popcount + branchless argmax (pack agreement<<12 | case, take
    // max). `sum`/`sumsq` accumulate the group's agreement distribution for the z-score
    // self-weight (empirical variance — the analytic closed form loses too much).
    let track = conf > 0;
    let mut best_key = vec![0u32; a_len * na];
    let mut sum = vec![0u64; if track { a_len * na } else { 0 }];
    let mut sumsq = vec![0u64; if track { a_len * na } else { 0 }];
    let mut count = 0u64;
    for j in 0..f.n_tok {
        if tok_key[j] == exclude {
            continue;
        }
        let wb = &f.words[j * w..j * w + w];
        let cj = f.tok_case[j];
        count += 1;
        for qi in 0..a_len {
            let wa = &qbuf[qi * w..qi * w + w];
            let row = qi * na;
            for gi in 0..na {
                let base = starts[gi];
                // Const group widths so the inner popcount unrolls & vectorizes
                // (gw=8 full-head groups, gw=2 head-folded groups).
                let ag = if gw == 8 {
                    let mut a = 0u32;
                    for k in 0..8 {
                        a += (!(wa[base + k] ^ wb[base + k])).count_ones();
                    }
                    a
                } else if gw == 2 {
                    (!(wa[base] ^ wb[base])).count_ones()
                        + (!(wa[base + 1] ^ wb[base + 1])).count_ones()
                } else {
                    let mut a = 0u32;
                    for k in 0..gw {
                        a += (!(wa[base + k] ^ wb[base + k])).count_ones();
                    }
                    a
                };
                let idx = row + gi;
                let key = (ag << 12) | cj;
                if key > best_key[idx] {
                    best_key[idx] = key;
                }
                if track {
                    sum[idx] += ag as u64;
                    sumsq[idx] += (ag as u64) * (ag as u64);
                }
            }
        }
    }
    let cf = count.max(1) as f32;
    let mut votes = vec![0f32; f.n_cases];
    for qi in 0..a_len {
        for (gi, &(_, wt)) in active.iter().enumerate() {
            let idx = qi * na + gi;
            let key = best_key[idx];
            if key == 0 {
                continue;
            }
            let bc = (key & 0xFFF) as usize;
            let best = (key >> 12) as f32;
            let strength = if track {
                let mean = sum[idx] as f32 / cf;
                match conf {
                    1 => (best - mean).max(0.0),
                    _ => {
                        let var = (sumsq[idx] as f32 / cf - mean * mean).max(1e-6);
                        ((best - mean) / var.sqrt()).max(0.0)
                    }
                }
            } else {
                1.0
            };
            votes[bc] += wt * strength;
        }
    }
    let mut order: Vec<usize> = (0..f.n_cases).filter(|&b| case_key[b] != exclude).collect();
    order.sort_by(|&x, &y| {
        votes[y]
            .partial_cmp(&votes[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let st = |b: usize| &f.case_tool[b] == tool_a;
    let sf = |b: usize| stem(&f.case_tool[b]) == stem(tool_a);
    Some([
        st(order[0]),
        order.iter().take(5).any(|&b| st(b)),
        sf(order[0]),
        order.iter().take(5).any(|&b| sf(b)),
    ])
}

fn pct(res: &[[bool; 4]], k: usize) -> f64 {
    let n = res.len().max(1);
    100.0 * res.iter().filter(|r| r[k]).count() as f64 / n as f64
}

/// Per-token/per-case fold keys for §78 k-fold: each conversation's round-robin
/// position mod `k`, projected onto tokens. Returns `(case_fold, tok_fold)`.
fn fold_keys(f: &Flat, conv_pos: &HashMap<u64, usize>, k: usize) -> (Vec<u32>, Vec<u32>) {
    let case_fold: Vec<u32> = f
        .case_conv
        .iter()
        .map(|cv| (conv_pos[cv] % k) as u32)
        .collect();
    let n_gconv = f.case_gconv.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut gconv_fold = vec![0u32; n_gconv];
    for (ci, &gc) in f.case_gconv.iter().enumerate() {
        gconv_fold[gc as usize] = case_fold[ci];
    }
    let tok_fold: Vec<u32> = f
        .tok_gconv
        .iter()
        .map(|&gc| gconv_fold[gc as usize])
        .collect();
    (case_fold, tok_fold)
}

/// §78 k-fold scan (flat concat vote).
fn run_k(f: &Flat, conv_pos: &HashMap<u64, usize>, k: usize) -> Vec<[bool; 4]> {
    let (case_fold, tok_fold) = fold_keys(f, conv_pos, k);
    (0..f.n_cases)
        .into_par_iter()
        .filter_map(|a| score_case(f, a, &tok_fold, &case_fold, case_fold[a]))
        .collect()
}

/// §78 k-fold scan with per-group weights (each group `gw` words). `late` selects the
/// vote-tally fusion (per-group argmax then combine) vs. early fusion (combine then
/// argmax).
fn run_k_weighted(
    f: &Flat,
    conv_pos: &HashMap<u64, usize>,
    k: usize,
    weights: &[f32],
    gw: usize,
    late: bool,
    conf: u8,
) -> Vec<[bool; 4]> {
    let (case_fold, tok_fold) = fold_keys(f, conv_pos, k);
    let freq: Vec<f32> = Vec::new();
    (0..f.n_cases)
        .into_par_iter()
        .filter_map(|a| {
            if late {
                score_case_late(
                    f,
                    a,
                    &tok_fold,
                    &case_fold,
                    case_fold[a],
                    weights,
                    gw,
                    conf,
                    &freq,
                )
            } else {
                score_case_weighted(f, a, &tok_fold, &case_fold, case_fold[a], weights, gw)
            }
        })
        .collect()
}

/// Parse `LFOLD_SWEEP` as either an inclusive range `"1-24"` or a comma list `"1,2,4"`.
fn parse_sweep(s: &str) -> Vec<usize> {
    if let Some((a, b)) = s.split_once('-') {
        if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
            return (a..=b).collect();
        }
    }
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let back: usize = std::env::var("BACK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let lfold: usize = std::env::var("LFOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let shift: usize = std::env::var("SHIFT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // LSEL=a-b keeps only layers [a, b) before folding (test which layers carry signal).
    let lsel: Option<(usize, usize)> = std::env::var("LSEL").ok().and_then(|s| {
        s.split_once('-')
            .and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?)))
    });

    let t = Instant::now();
    let mut substrate = Substrate::new();
    let _p = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open: {e}"))?;
    eprintln!("substrate open      : {:.2?}", t.elapsed());

    let t = Instant::now();
    let cases = projection_probe_cases(&substrate, back);
    // Round-robin each tool's conversations across folds (settled tool per conv) — the
    // fold assignment is LFOLD-independent, so compute it once.
    let mut conv_tool: HashMap<u64, String> = HashMap::new();
    for c in &cases {
        conv_tool.insert(c.conv, c.tool.clone());
    }
    let mut convs_by_tool: HashMap<&str, Vec<u64>> = HashMap::new();
    for (cv, tl) in &conv_tool {
        convs_by_tool.entry(tl.as_str()).or_default().push(*cv);
    }
    let mut conv_pos: HashMap<u64, usize> = HashMap::new();
    for cvs in convs_by_tool.values_mut() {
        cvs.sort_unstable();
        for (p, cv) in cvs.iter().enumerate() {
            conv_pos.insert(*cv, p);
        }
    }
    let n_convs = conv_tool.len();
    eprintln!(
        "extract             : {:.2?}   ({} probes, {} conversations)",
        t.elapsed(),
        cases.len(),
        n_convs
    );

    // ── ORG: structural breakdown of the ProbeCase collection ──
    if std::env::var("ORG").is_ok() {
        let mut per_tool: HashMap<&str, usize> = HashMap::new();
        let mut per_conv: HashMap<u64, usize> = HashMap::new();
        let mut per_tool_convs: HashMap<&str, std::collections::HashSet<u64>> = HashMap::new();
        for c in &cases {
            *per_tool.entry(c.tool.as_str()).or_default() += 1;
            *per_conv.entry(c.conv).or_default() += 1;
            per_tool_convs
                .entry(c.tool.as_str())
                .or_default()
                .insert(c.conv);
        }
        println!("\n══ ProbeCase collection ══");
        println!("total ProbeCases : {}", cases.len());
        println!("distinct tools   : {}", per_tool.len());
        println!("distinct convs   : {}", per_conv.len());

        // cases-per-conv distribution
        let mut pc: Vec<usize> = per_conv.values().copied().collect();
        pc.sort_unstable();
        let sum: usize = pc.iter().sum();
        println!(
            "cases per conv   : min {} p50 {} max {} mean {:.2}",
            pc[0],
            pc[pc.len() / 2],
            pc[pc.len() - 1],
            sum as f64 / pc.len() as f64
        );
        let mut cpc: HashMap<usize, usize> = HashMap::new();
        for &v in &pc {
            *cpc.entry(v).or_default() += 1;
        }
        let mut cpc: Vec<(usize, usize)> = cpc.into_iter().collect();
        cpc.sort_unstable();
        println!("  histogram (cases-per-conv → #convs):");
        for (v, n) in &cpc {
            println!("    {:>3} cases : {:>4} convs", v, n);
        }

        // per-tool: #convs and #cases
        let mut tools: Vec<(&str, usize, usize)> = per_tool
            .iter()
            .map(|(t, &cnt)| (*t, per_tool_convs[t].len(), cnt))
            .collect();
        tools.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(b.0)));
        println!("\n  per-tool (tool → #convs, #cases):");
        for (t, nc, ncase) in tools.iter().take(40) {
            println!("    {:>4} cases  {:>3} convs   {}", ncase, nc, t);
        }
        if tools.len() > 40 {
            println!("    … {} more tools", tools.len() - 40);
        }
        return Ok(());
    }

    // ── §80.1 MECH: belief-update mechanism sweep on single-intent + topic-switch ──
    // Same LOO per-projection scores, but evaluated two ways: (1) single-intent — walk one
    // conversation, predict its tool; (2) topic-switch — concatenate a different-tool conv A
    // BEFORE conv B and predict B, so a good mechanism must let B override A's stale belief.
    // A generic winner scores high on BOTH (we rank by min of the two Tool-1 rates).
    if std::env::var("MECH").is_ok() {
        let (n_tools, case_tool, stem_id) = build_tool_index(&cases);
        let case_conv: Vec<u64> = cases.iter().map(|c| c.conv).collect();
        let t = Instant::now();
        let proj_score = precompute_proj_scores(&cases, &case_conv, &case_tool, n_tools);
        eprintln!("MECH precompute: {:.2?}", t.elapsed());

        // Calibrate score magnitude so absolute-scale mechs get sane grids.
        let mut vals: Vec<f32> = proj_score
            .iter()
            .flat_map(|v| v.iter().copied())
            .filter(|&x| x > 0.0)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let qp = |q: f64| vals[((vals.len() - 1) as f64 * q) as usize];
        eprintln!(
            "score nonzero: p50 {:.2}  p95 {:.2}  max {:.2}",
            qp(0.5),
            qp(0.95),
            vals[vals.len() - 1]
        );

        // Group projections by conversation, decode order.
        let mut by_conv: HashMap<u64, Vec<usize>> = HashMap::new();
        for (ci, &cv) in case_conv.iter().enumerate() {
            by_conv.entry(cv).or_default().push(ci);
        }
        for v in by_conv.values_mut() {
            v.sort_by_key(|&ci| cases[ci].point);
        }
        let mut convs: Vec<u64> = by_conv.keys().copied().collect();
        convs.sort_unstable();
        let n = convs.len();

        // Regime 1 — single-intent: (walk = conv's projections, truth = its tool).
        let single: Vec<(Vec<usize>, usize)> = convs
            .iter()
            .map(|cv| {
                let projs = by_conv[cv].clone();
                let truth = case_tool[*projs.last().unwrap()];
                (projs, truth)
            })
            .collect();

        // Regime 2 — topic-switch: A (different tool) walked BEFORE B, predict B. Pair each
        // B with A's at spread offsets so the switch spans many tool pairs.
        let offs = [1usize, n / 4, n / 2, 3 * n / 4];
        let mut switch: Vec<(Vec<usize>, usize)> = Vec::new();
        for (i, &b) in convs.iter().enumerate() {
            let b_projs = &by_conv[&b];
            let b_tool = case_tool[b_projs[0]];
            for &off in &offs {
                let a = convs[(i + off) % n];
                if a == b || case_tool[by_conv[&a][0]] == b_tool {
                    continue;
                }
                let mut walk = by_conv[&a].clone();
                walk.extend_from_slice(b_projs);
                switch.push((walk, b_tool));
            }
        }

        let eval = |trials: &[(Vec<usize>, usize)], mech: &Mech| -> [f64; 4] {
            let mut agg = [0usize; 4];
            for (projs, truth) in trials {
                let acc = walk_final(projs, &proj_score, n_tools, mech);
                let mut order: Vec<usize> = (0..n_tools).collect();
                order.sort_by(|&x, &y| {
                    acc[y]
                        .partial_cmp(&acc[x])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if order[0] == *truth {
                    agg[0] += 1;
                }
                if order.iter().take(5).any(|&tt| tt == *truth) {
                    agg[1] += 1;
                }
                if stem_id[order[0]] == stem_id[*truth] {
                    agg[2] += 1;
                }
                if order
                    .iter()
                    .take(5)
                    .any(|&tt| stem_id[tt] == stem_id[*truth])
                {
                    agg[3] += 1;
                }
            }
            let m = trials.len().max(1) as f64;
            [
                100.0 * agg[0] as f64 / m,
                100.0 * agg[1] as f64 / m,
                100.0 * agg[2] as f64 / m,
                100.0 * agg[3] as f64 / m,
            ]
        };

        // Mechanism roster.
        let mut mechs: Vec<(String, Mech)> = vec![
            ("Sum (pool, no decay)".into(), Mech::Sum),
            ("Max (pool, no decay)".into(), Mech::Max),
        ];
        for &l in &[0.99f32, 0.95, 0.9, 0.7, 0.5, 0.3] {
            mechs.push((format!("Mult λ={l:.2}"), Mech::Mult(l)));
        }
        for &l in &[0.99f32, 0.95, 0.9, 0.85, 0.7, 0.5] {
            mechs.push((format!("LeakyMax λ={l:.2}"), Mech::LeakyMax(l)));
        }
        for &b in &[0.2f32, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] {
            mechs.push((format!("RelLeak β={b:.2}"), Mech::RelLeak(b)));
        }
        for &g in &[0.25f32, 0.5, 1.0, 2.0] {
            mechs.push((format!("Ramp g={g:.2}"), Mech::Ramp(g)));
        }
        for &l in &[0.5f32, 0.7, 0.9, 0.95] {
            mechs.push((format!("Simplex λ={l:.2}"), Mech::Simplex(l)));
        }
        for &(lf, a) in &[(0.5f32, 0.3f32), (0.5, 1.0), (0.7, 0.5), (0.3, 0.5)] {
            mechs.push((
                format!("TwoScale λf={lf:.1} α={a:.1}"),
                Mech::TwoScale(lf, a),
            ));
        }
        for &d in &[0.0f32, 0.1, 0.2, 0.35, 0.5] {
            mechs.push((format!("Surprise d={d:.2}"), Mech::Surprise(d)));
        }
        for &d in &[0.0f32, 0.1, 0.2, 0.35, 0.5] {
            mechs.push((format!("SurpMax d={d:.2}"), Mech::SurpMax(d)));
        }

        println!("\n══ §80.1 — belief-update mechanism sweep ══");
        println!(
            "single-intent: {} convs · topic-switch: {} A→B pairs (predict B)",
            single.len(),
            switch.len()
        );
        println!(
            "  {:<24} {:>6} {:>6}   {:>6} {:>6}   {:>7}",
            "mechanism", "S-T1", "S-T5", "W-T1", "W-T5", "min-T1"
        );
        let mut rows: Vec<(String, [f64; 4], [f64; 4])> = mechs
            .iter()
            .map(|(name, m)| (name.clone(), eval(&single, m), eval(&switch, m)))
            .collect();
        // Print in roster order, then a ranked-by-min-T1 leaderboard.
        for (name, s, w) in &rows {
            println!(
                "  {:<24} {:>6.1} {:>6.1}   {:>6.1} {:>6.1}   {:>7.1}",
                name,
                s[0],
                s[1],
                w[0],
                w[1],
                s[0].min(w[0])
            );
        }
        rows.sort_by(|a, b| {
            (b.1[0].min(b.2[0]))
                .partial_cmp(&(a.1[0].min(a.2[0])))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!("\n  top 6 by min(S-T1, W-T1):");
        for (name, s, w) in rows.iter().take(6) {
            println!(
                "  {:<24} S-T1 {:>5.1}  W-T1 {:>5.1}  min {:>5.1}",
                name,
                s[0],
                w[0],
                s[0].min(w[0])
            );
        }
        return Ok(());
    }

    // ── §80.2 POLICY: selection-policy sweep (thresholds, budgets, per-slot β) ──
    // Wraps the locked RelLeak belief in the production SectionSelector and sweeps its
    // knobs on the same single-intent + topic-switch regimes. Reports, per policy: top-1
    // of the SELECTED set, recall (truth in the budgeted set), and average set size — so
    // we can pick values for the zend substrate template. In the topic-switch regime the
    // online belief simply decays across the A→B boundary (no per-turn pin).
    if std::env::var("POLICY").is_ok() {
        let (n_tools, case_tool, _stem_id) = build_tool_index(&cases);
        let case_conv: Vec<u64> = cases.iter().map(|c| c.conv).collect();
        let t = Instant::now();
        let proj_score = precompute_proj_scores(&cases, &case_conv, &case_tool, n_tools);
        eprintln!("POLICY precompute: {:.2?}", t.elapsed());

        let mut by_conv: HashMap<u64, Vec<usize>> = HashMap::new();
        for (ci, &cv) in case_conv.iter().enumerate() {
            by_conv.entry(cv).or_default().push(ci);
        }
        for v in by_conv.values_mut() {
            v.sort_by_key(|&ci| cases[ci].point);
        }
        let mut convs: Vec<u64> = by_conv.keys().copied().collect();
        convs.sort_unstable();
        let n = convs.len();
        let single: Vec<(Vec<usize>, usize)> = convs
            .iter()
            .map(|cv| {
                let projs = by_conv[cv].clone();
                (projs.clone(), case_tool[*projs.last().unwrap()])
            })
            .collect();
        let offs = [1usize, n / 4, n / 2, 3 * n / 4];
        let mut switch: Vec<(Vec<usize>, Vec<usize>, usize)> = Vec::new();
        for (i, &b) in convs.iter().enumerate() {
            let b_projs = &by_conv[&b];
            let b_tool = case_tool[b_projs[0]];
            for &off in &offs {
                let a = convs[(i + off) % n];
                if a == b || case_tool[by_conv[&a][0]] == b_tool {
                    continue;
                }
                switch.push((by_conv[&a].clone(), b_projs.clone(), b_tool));
            }
        }

        // Calibrate: final belief leader / 5th-place scores (RelLeak β=0.40, single-intent).
        {
            let mut leader = Vec::new();
            let mut fifth = Vec::new();
            for (projs, _) in &single {
                let acc = walk_final(projs, &proj_score, n_tools, &Mech::RelLeak(0.40));
                let mut v = acc.clone();
                v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                leader.push(v[0]);
                fifth.push(*v.get(4).unwrap_or(&0.0));
            }
            leader.sort_by(|a, b| a.partial_cmp(b).unwrap());
            fifth.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |v: &[f32], p: f64| v[((v.len() - 1) as f64 * p) as usize];
            eprintln!(
                "final belief (β=0.40): leader p10 {:.1} p50 {:.1} p90 {:.1} | 5th p50 {:.1} p90 {:.1}",
                q(&leader, 0.1),
                q(&leader, 0.5),
                q(&leader, 0.9),
                q(&fifth, 0.5),
                q(&fifth, 0.9),
            );
        }

        // Evaluate one policy preset. All tools sit in one budget group.
        // Returns [tool-1%, recall%, avg-selected] for a regime.
        let run = |beta: f32,
                   min_s: f32,
                   evict_s: f32,
                   budget: GroupBudget,
                   switch_regime: bool|
         -> [f64; 3] {
            let pol = SectionPolicy {
                group: 0,
                beta,
                min_score: min_s,
                evict_score: evict_s,
            };
            let (mut t1, mut rec, mut ssz, mut cnt) = (0usize, 0usize, 0usize, 0usize);
            let mut eval = |walk_a: &[usize], walk_b: Option<&[usize]>, truth: usize| {
                let mut sel = SectionSelector::new(vec![pol; n_tools], vec![budget]);
                for &pj in walk_a {
                    sel.update(&proj_score[pj]);
                }
                if let Some(wb) = walk_b {
                    // Topic switch: the online belief simply decays across the
                    // boundary — no pin to release.
                    for &pj in wb {
                        sel.update(&proj_score[pj]);
                    }
                }
                let chosen = sel.selected_slots();
                if sel.top_selected() == Some(truth) {
                    t1 += 1;
                }
                if chosen.contains(&truth) {
                    rec += 1;
                }
                ssz += chosen.len();
                cnt += 1;
            };
            if switch_regime {
                for (a, b, truth) in &switch {
                    eval(a, Some(b), *truth);
                }
            } else {
                for (a, truth) in &single {
                    eval(a, None, *truth);
                }
            }
            let m = cnt.max(1) as f64;
            [
                100.0 * t1 as f64 / m,
                100.0 * rec as f64 / m,
                ssz as f64 / m,
            ]
        };

        println!("\n══ §80.2 — selection-policy sweep (β=0.40, group min=1) ══");
        println!(
            "single {} convs · switch {} A→B pairs · Tool-1 = top of selected set, Rec = truth in set",
            single.len(),
            switch.len()
        );
        println!(
            "  {:<34} {:>6} {:>6} {:>5}   {:>6} {:>6} {:>5}",
            "policy (min/evict, max)", "S-T1", "S-Rec", "S-sz", "W-T1", "W-Rec", "W-sz"
        );
        let presets: [(f32, f32, usize); 10] = [
            (0.0, 0.0, 1),
            (0.0, 0.0, 3),
            (0.0, 0.0, 5),
            (15.0, 8.0, 5),
            (25.0, 12.0, 5),
            (40.0, 20.0, 5),
            (25.0, 12.0, 3),
            (40.0, 20.0, 3),
            (25.0, 12.0, 1),
            (40.0, 20.0, 1),
        ];
        for (min_s, evict_s, max) in presets {
            let budget = GroupBudget { min: 1, max };
            let s = run(0.40, min_s, evict_s, budget, false);
            let w = run(0.40, min_s, evict_s, budget, true);
            let tag = format!("min{:.0}/ev{:.0} max{}", min_s, evict_s, max);
            println!(
                "  {:<34} {:>6.1} {:>6.1} {:>5.2}   {:>6.1} {:>6.1} {:>5.2}",
                tag, s[0], s[1], s[2], w[0], w[1], w[2]
            );
        }
        return Ok(());
    }

    // ── §80: leave-one-conversation-out online decaying tool belief ──
    // Holdout = one conversation. Walk its projections in decode order; each scans the
    // full gallery (all OTHER conversations) → per-tool confidence (sum-of-z). Maintain a
    // running tool→confidence belief that DECAYS the past each step and merges the fresh
    // scores, so recent sharp projections dominate stale vague ones. Prediction at the end
    // of the walk = top tool. Compared against last-only and uniform-sum baselines.
    if std::env::var("WALK").is_ok() {
        let (n_tools, case_tool, stem_id) = build_tool_index(&cases);
        let case_conv: Vec<u64> = cases.iter().map(|c| c.conv).collect();
        let t = Instant::now();
        let proj_score = precompute_proj_scores(&cases, &case_conv, &case_tool, n_tools);
        eprintln!(
            "WALK precompute: {:.2?}  ({} projections, {} tools)",
            t.elapsed(),
            cases.len(),
            n_tools
        );

        // Group projections by conversation, ordered by decode point.
        let mut by_conv: HashMap<u64, Vec<usize>> = HashMap::new();
        for (ci, &cv) in case_conv.iter().enumerate() {
            by_conv.entry(cv).or_default().push(ci);
        }
        for v in by_conv.values_mut() {
            v.sort_by_key(|&ci| cases[ci].point);
        }
        let mut mixed = 0usize;
        for v in by_conv.values() {
            let t0 = case_tool[v[0]];
            if v.iter().any(|&ci| case_tool[ci] != t0) {
                mixed += 1;
            }
        }
        let convs: Vec<u64> = {
            let mut c: Vec<u64> = by_conv.keys().copied().collect();
            c.sort_unstable();
            c
        };

        // Score one belief-update policy over all holdout conversations.
        // merge: 0=accumulate (conf += score), 1=max (conf = max(decayed, score)).
        // per_token: decay by token gap between projections rather than one step.
        let run = |lambda: f32, merge: u8, per_token: bool| -> [f64; 4] {
            let mut agg = [0usize; 4];
            for &cv in &convs {
                let projs = &by_conv[&cv];
                let truth = case_tool[*projs.last().unwrap()];
                let mut acc = vec![0f32; n_tools];
                let mut prev_pt = cases[projs[0]].point;
                for (i, &pj) in projs.iter().enumerate() {
                    let d = if i == 0 {
                        1.0
                    } else if per_token {
                        lambda.powi((cases[pj].point - prev_pt) as i32)
                    } else {
                        lambda
                    };
                    prev_pt = cases[pj].point;
                    if i > 0 {
                        for x in acc.iter_mut() {
                            *x *= d;
                        }
                    }
                    let s = &proj_score[pj];
                    match merge {
                        1 => {
                            for t in 0..n_tools {
                                acc[t] = acc[t].max(s[t]);
                            }
                        }
                        _ => {
                            for t in 0..n_tools {
                                acc[t] += s[t];
                            }
                        }
                    }
                }
                let mut order: Vec<usize> = (0..n_tools).collect();
                order.sort_by(|&x, &y| {
                    acc[y]
                        .partial_cmp(&acc[x])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let st = |t: usize| t == truth;
                let sf = |t: usize| stem_id[t] == stem_id[truth];
                if st(order[0]) {
                    agg[0] += 1;
                }
                if order.iter().take(5).any(|&t| st(t)) {
                    agg[1] += 1;
                }
                if sf(order[0]) {
                    agg[2] += 1;
                }
                if order.iter().take(5).any(|&t| sf(t)) {
                    agg[3] += 1;
                }
            }
            let n = convs.len() as f64;
            [
                100.0 * agg[0] as f64 / n,
                100.0 * agg[1] as f64 / n,
                100.0 * agg[2] as f64 / n,
                100.0 * agg[3] as f64 / n,
            ]
        };

        // last-only baseline: use only the final projection's scores.
        let last_only = {
            let mut agg = [0usize; 4];
            for &cv in &convs {
                let projs = &by_conv[&cv];
                let truth = case_tool[*projs.last().unwrap()];
                let s = &proj_score[*projs.last().unwrap()];
                let mut order: Vec<usize> = (0..n_tools).collect();
                order
                    .sort_by(|&x, &y| s[y].partial_cmp(&s[x]).unwrap_or(std::cmp::Ordering::Equal));
                let st = |t: usize| t == truth;
                let sf = |t: usize| stem_id[t] == stem_id[truth];
                if st(order[0]) {
                    agg[0] += 1;
                }
                if order.iter().take(5).any(|&t| st(t)) {
                    agg[1] += 1;
                }
                if sf(order[0]) {
                    agg[2] += 1;
                }
                if order.iter().take(5).any(|&t| sf(t)) {
                    agg[3] += 1;
                }
            }
            let n = convs.len() as f64;
            [
                100.0 * agg[0] as f64 / n,
                100.0 * agg[1] as f64 / n,
                100.0 * agg[2] as f64 / n,
                100.0 * agg[3] as f64 / n,
            ]
        };

        println!("\n══ §80 — leave-one-conversation-out · online decaying tool belief ══");
        println!(
            "{} holdout conversations · {} tools · {} mixed-tool convs",
            convs.len(),
            n_tools,
            mixed
        );
        let row = |tag: &str, m: [f64; 4]| {
            println!(
                "  {:<22} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                tag, m[0], m[1], m[2], m[3]
            );
        };
        println!(
            "  {:<22} {:>7} {:>7} {:>7} {:>7}",
            "policy", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        row("last-only", last_only);
        row("uniform (λ=1 acc)", run(1.0, 0, false));
        println!("  ── per-step decay · accumulate ──");
        for &l in &[0.9f32, 0.7, 0.5, 0.3, 0.1] {
            row(&format!("λ={l:.2} step acc"), run(l, 0, false));
        }
        println!("  ── per-step decay · max-merge ──");
        for &l in &[1.0f32, 0.9, 0.7, 0.5, 0.3] {
            row(&format!("λ={l:.2} step max"), run(l, 1, false));
        }
        println!("  ── per-token decay · accumulate ──");
        for &l in &[0.999f32, 0.995, 0.99, 0.98, 0.95] {
            row(&format!("λ={l:.3} tok acc"), run(l, 0, true));
        }
        return Ok(());
    }

    // ── MAXCONF: pick the projection's selection from the highest-confidence token ──
    // Baseline ranks cases by the SUM of every token's z-votes. This ranks cases by the
    // MAX confidence of any single token pointing at them — "walk the probe, let the most
    // confident token drive the selection." Both computed in one pass for a fair compare.
    if std::env::var("MAXCONF").is_ok() {
        let k = 4usize;
        let case_fold: Vec<usize> = cases.iter().map(|c| conv_pos[&c.conv] % k).collect();
        let n_cases = cases.len();
        let mut gal: Vec<(Vec<&WideQSig>, Vec<u32>)> =
            (0..k).map(|_| (Vec::new(), Vec::new())).collect();
        for (ci, c) in cases.iter().enumerate() {
            let cf = case_fold[ci];
            for (f, g) in gal.iter_mut().enumerate() {
                if f != cf {
                    for tok in &c.window {
                        g.0.push(tok);
                        g.1.push(ci as u32);
                    }
                }
            }
        }
        let n_groups = cases
            .iter()
            .flat_map(|c| c.window.first())
            .map(|t| t.n_heads as usize / 4)
            .next()
            .unwrap_or(3);
        let gw = 8usize;

        let t = Instant::now();
        // Per probe: return (sum-z metrics, max-conf metrics).
        let res: Vec<([bool; 4], [bool; 4])> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                let fa = case_fold[a];
                if !(0..n_cases).any(|b| case_fold[b] != fa && cases[b].tool == cases[a].tool) {
                    return None;
                }
                let (gtok, gcase) = &gal[fa];
                let n_gal = gtok.len().max(1) as f32;
                let mut votes_sum = vec![0f32; n_cases];
                let mut case_maxconf = vec![0f32; n_cases];
                for q in &cases[a].window {
                    let mut tok_votes: Vec<(usize, f32)> = Vec::with_capacity(n_groups);
                    for g in 0..n_groups {
                        let base = g * gw;
                        let qg = &q.words[base..base + gw];
                        let (mut best, mut bidx) = (0u32, usize::MAX);
                        let (mut sum, mut sumsq) = (0u64, 0u64);
                        for (j, cand) in gtok.iter().enumerate() {
                            let cw = &cand.words[base..base + gw];
                            let mut ag = 0u32;
                            for kk in 0..8 {
                                ag += (!(qg[kk] ^ cw[kk])).count_ones();
                            }
                            if ag > best {
                                best = ag;
                                bidx = j;
                            }
                            sum += ag as u64;
                            sumsq += (ag as u64) * (ag as u64);
                        }
                        if bidx != usize::MAX {
                            let mean = sum as f32 / n_gal;
                            let var = (sumsq as f32 / n_gal - mean * mean).max(1e-6);
                            let z = ((best as f32 - mean) / var.sqrt()).max(0.0);
                            let c = gcase[bidx] as usize;
                            votes_sum[c] += z;
                            tok_votes.push((c, z));
                        }
                    }
                    // Token confidence = z-sum of its winning case; update per-case max.
                    let (mut pc, mut conf) = (usize::MAX, 0f32);
                    for &(c, _) in &tok_votes {
                        let total: f32 = tok_votes
                            .iter()
                            .filter(|&&(cc, _)| cc == c)
                            .map(|&(_, z)| z)
                            .sum();
                        if total > conf {
                            conf = total;
                            pc = c;
                        }
                    }
                    if pc != usize::MAX && conf > case_maxconf[pc] {
                        case_maxconf[pc] = conf;
                    }
                }
                let rank = |score: &[f32]| -> [bool; 4] {
                    let mut order: Vec<usize> =
                        (0..n_cases).filter(|&b| case_fold[b] != fa).collect();
                    order.sort_by(|&x, &y| {
                        score[y]
                            .partial_cmp(&score[x])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let st = |b: usize| cases[b].tool == cases[a].tool;
                    let sf = |b: usize| stem(&cases[b].tool) == stem(&cases[a].tool);
                    [
                        st(order[0]),
                        order.iter().take(5).any(|&b| st(b)),
                        sf(order[0]),
                        order.iter().take(5).any(|&b| sf(b)),
                    ]
                };
                Some((rank(&votes_sum), rank(&case_maxconf)))
            })
            .collect();
        eprintln!(
            "MAXCONF scan: {:.2?}  ({} scorable)",
            t.elapsed(),
            res.len()
        );
        let pc = |sel: usize, kk: usize| {
            let n = res.len().max(1);
            100.0
                * res
                    .iter()
                    .filter(|r| if sel == 0 { r.0[kk] } else { r.1[kk] })
                    .count() as f64
                / n as f64
        };
        println!("\n══ §79.1 — selection: sum-of-z vote  vs  max-confidence token ══");
        println!(
            "  {:<18} {:>7} {:>7} {:>7} {:>7}",
            "selection", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        println!(
            "  {:<18} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            "sum-of-z (base)",
            pc(0, 0),
            pc(0, 1),
            pc(0, 2),
            pc(0, 3)
        );
        println!(
            "  {:<18} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            "max-confidence",
            pc(1, 0),
            pc(1, 1),
            pc(1, 2),
            pc(1, 3)
        );
        return Ok(());
    }

    // ── CONFGATE: confidence-gated consensus — drop tokens below conf g, then sum-vote ─
    // Keeps aggregation (unlike MAXCONF) but removes the low-confidence noise majority
    // (conf-4/5 tokens hit only 8–20%). Sweep the gate to find where it beats 81.0.
    if std::env::var("CONFGATE").is_ok() {
        let k = 4usize;
        let case_fold: Vec<usize> = cases.iter().map(|c| conv_pos[&c.conv] % k).collect();
        let n_cases = cases.len();
        let mut gal: Vec<(Vec<&WideQSig>, Vec<u32>)> =
            (0..k).map(|_| (Vec::new(), Vec::new())).collect();
        for (ci, c) in cases.iter().enumerate() {
            let cf = case_fold[ci];
            for (f, g) in gal.iter_mut().enumerate() {
                if f != cf {
                    for tok in &c.window {
                        g.0.push(tok);
                        g.1.push(ci as u32);
                    }
                }
            }
        }
        let n_groups = cases
            .iter()
            .flat_map(|c| c.window.first())
            .map(|t| t.n_heads as usize / 4)
            .next()
            .unwrap_or(3);
        let gw = 8usize;
        let gates: Vec<f32> = (0..=16).map(|g| g as f32).collect();

        let t = Instant::now();
        // Per probe: for each gate, [Tool-1,Tool-5,Fam-1,Fam-5] + surviving-token count.
        let res: Vec<(Vec<[bool; 4]>, Vec<u32>)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                let fa = case_fold[a];
                if !(0..n_cases).any(|b| case_fold[b] != fa && cases[b].tool == cases[a].tool) {
                    return None;
                }
                let (gtok, gcase) = &gal[fa];
                let n_gal = gtok.len().max(1) as f32;
                // Per token: its group votes (case, z) and confidence (z-sum of winning case).
                let mut toks: Vec<(Vec<(usize, f32)>, f32)> =
                    Vec::with_capacity(cases[a].window.len());
                for q in &cases[a].window {
                    let mut tv: Vec<(usize, f32)> = Vec::with_capacity(n_groups);
                    for g in 0..n_groups {
                        let base = g * gw;
                        let qg = &q.words[base..base + gw];
                        let (mut best, mut bidx) = (0u32, usize::MAX);
                        let (mut sum, mut sumsq) = (0u64, 0u64);
                        for (j, cand) in gtok.iter().enumerate() {
                            let cw = &cand.words[base..base + gw];
                            let mut ag = 0u32;
                            for kk in 0..8 {
                                ag += (!(qg[kk] ^ cw[kk])).count_ones();
                            }
                            if ag > best {
                                best = ag;
                                bidx = j;
                            }
                            sum += ag as u64;
                            sumsq += (ag as u64) * (ag as u64);
                        }
                        if bidx != usize::MAX {
                            let mean = sum as f32 / n_gal;
                            let var = (sumsq as f32 / n_gal - mean * mean).max(1e-6);
                            let z = ((best as f32 - mean) / var.sqrt()).max(0.0);
                            tv.push((gcase[bidx] as usize, z));
                        }
                    }
                    let mut conf = 0f32;
                    for &(c, _) in &tv {
                        let total: f32 =
                            tv.iter().filter(|&&(cc, _)| cc == c).map(|&(_, z)| z).sum();
                        conf = conf.max(total);
                    }
                    toks.push((tv, conf));
                }
                let st = |b: usize| cases[b].tool == cases[a].tool;
                let sf = |b: usize| stem(&cases[b].tool) == stem(&cases[a].tool);
                let mut per_gate = Vec::with_capacity(gates.len());
                let mut surv = Vec::with_capacity(gates.len());
                for &gate in &gates {
                    let mut votes = vec![0f32; n_cases];
                    let mut n = 0u32;
                    for (tv, conf) in &toks {
                        if *conf >= gate {
                            n += 1;
                            for &(c, z) in tv {
                                votes[c] += z;
                            }
                        }
                    }
                    let mut order: Vec<usize> =
                        (0..n_cases).filter(|&b| case_fold[b] != fa).collect();
                    order.sort_by(|&x, &y| {
                        votes[y]
                            .partial_cmp(&votes[x])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    per_gate.push([
                        st(order[0]),
                        order.iter().take(5).any(|&b| st(b)),
                        sf(order[0]),
                        order.iter().take(5).any(|&b| sf(b)),
                    ]);
                    surv.push(n);
                }
                Some((per_gate, surv))
            })
            .collect();
        eprintln!(
            "CONFGATE scan: {:.2?}  ({} scorable)",
            t.elapsed(),
            res.len()
        );
        let n = res.len().max(1) as f64;
        println!(
            "\n══ §79.2 — confidence-gated consensus (drop tokens below gate, then sum-vote) ══"
        );
        println!(
            "  {:>4} {:>7} {:>7} {:>7} {:>7} {:>10}",
            "gate", "Tool-1", "Tool-5", "Fam-1", "Fam-5", "avg-surv"
        );
        for (gi, &gate) in gates.iter().enumerate() {
            let pc = |kk: usize| 100.0 * res.iter().filter(|r| r.0[gi][kk]).count() as f64 / n;
            let avg_surv = res.iter().map(|r| r.1[gi] as f64).sum::<f64>() / n;
            let tag = if gate == 0.0 { "  (=base)" } else { "" };
            println!(
                "  {:>4.0} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>10.1}{}",
                gate,
                pc(0),
                pc(1),
                pc(2),
                pc(3),
                avg_surv,
                tag
            );
        }
        return Ok(());
    }

    // ── CONFPROJ: per-PROJECTION confidence + per-LINK confidence for each turn it selects ─
    // Aggregates the sum-of-z vote per candidate turn (the proven baseline ranking), then
    // attaches a confidence to the projection AND to each turn/section it links to:
    //   • per-link  conf = votes[turn] / n_tokens   (avg per-token agreement toward that turn;
    //                                                 absolute → comparable across projections)
    //   • per-proj trust = winner's per-link conf   (how strong the top selection is)
    //   • per-proj margin = (winner − runner-up) / winner  (how clear-cut the top-1 is)
    // We bin projections by trust and by margin and report top-1 hit/family rate per bin.
    if std::env::var("CONFPROJ").is_ok() {
        let k = 4usize;
        let case_fold: Vec<usize> = cases.iter().map(|c| conv_pos[&c.conv] % k).collect();
        let n_cases = cases.len();
        let mut gal: Vec<(Vec<&WideQSig>, Vec<u32>)> =
            (0..k).map(|_| (Vec::new(), Vec::new())).collect();
        for (ci, c) in cases.iter().enumerate() {
            let cf = case_fold[ci];
            for (f, g) in gal.iter_mut().enumerate() {
                if f != cf {
                    for tok in &c.window {
                        g.0.push(tok);
                        g.1.push(ci as u32);
                    }
                }
            }
        }
        let n_groups = cases
            .iter()
            .flat_map(|c| c.window.first())
            .map(|t| t.n_heads as usize / 4)
            .next()
            .unwrap_or(3);
        let gw = 8usize;

        struct ProjConf {
            hit: bool,
            fam: bool,
            trust: f32,
            margin: f32,
            links: Vec<(String, u64, f32, bool)>, // (tool, turn/conv id, per-link conf, correct)
        }

        let t = Instant::now();
        let res: Vec<ProjConf> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                let fa = case_fold[a];
                if !(0..n_cases).any(|b| case_fold[b] != fa && cases[b].tool == cases[a].tool) {
                    return None;
                }
                let (gtok, gcase) = &gal[fa];
                let n_gal = gtok.len().max(1) as f32;
                let n_tok = cases[a].window.len().max(1) as f32;
                let mut votes = vec![0f32; n_cases];
                for q in &cases[a].window {
                    for g in 0..n_groups {
                        let base = g * gw;
                        let qg = &q.words[base..base + gw];
                        let (mut best, mut bidx) = (0u32, usize::MAX);
                        let (mut sum, mut sumsq) = (0u64, 0u64);
                        for (j, cand) in gtok.iter().enumerate() {
                            let cw = &cand.words[base..base + gw];
                            let mut ag = 0u32;
                            for kk in 0..8 {
                                ag += (!(qg[kk] ^ cw[kk])).count_ones();
                            }
                            if ag > best {
                                best = ag;
                                bidx = j;
                            }
                            sum += ag as u64;
                            sumsq += (ag as u64) * (ag as u64);
                        }
                        if bidx != usize::MAX {
                            let mean = sum as f32 / n_gal;
                            let var = (sumsq as f32 / n_gal - mean * mean).max(1e-6);
                            let z = ((best as f32 - mean) / var.sqrt()).max(0.0);
                            votes[gcase[bidx] as usize] += z;
                        }
                    }
                }
                let mut order: Vec<usize> = (0..n_cases).filter(|&b| case_fold[b] != fa).collect();
                order.sort_by(|&x, &y| {
                    votes[y]
                        .partial_cmp(&votes[x])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let win = order[0];
                let run = *order.get(1).unwrap_or(&win);
                let trust = votes[win] / n_tok;
                let margin = if votes[win] > 0.0 {
                    (votes[win] - votes[run]) / votes[win]
                } else {
                    0.0
                };
                let st = |b: usize| cases[b].tool == cases[a].tool;
                let sf = |b: usize| stem(&cases[b].tool) == stem(&cases[a].tool);
                let links = order
                    .iter()
                    .take(5)
                    .map(|&b| {
                        (
                            cases[b].tool.clone(),
                            cases[b].conv,
                            votes[b] / n_tok,
                            st(b),
                        )
                    })
                    .collect();
                Some(ProjConf {
                    hit: st(win),
                    fam: sf(win),
                    trust,
                    margin,
                    links,
                })
            })
            .collect();
        eprintln!(
            "CONFPROJ scan: {:.2?}  ({} scorable)",
            t.elapsed(),
            res.len()
        );
        let n = res.len().max(1) as f64;

        // Distribution of the per-projection trust score.
        let mut ts: Vec<f32> = res.iter().map(|r| r.trust).collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| ts[((ts.len() - 1) as f64 * p) as usize];
        println!("\n══ §79.3 — per-projection confidence + per-link confidence ══");
        println!(
            "gallery {} probes · vote scale = avg per-token z toward a turn",
            res.len()
        );
        println!(
            "trust percentiles: p10 {:.2}  p25 {:.2}  p50 {:.2}  p75 {:.2}  p90 {:.2}  max {:.2}",
            q(0.10),
            q(0.25),
            q(0.50),
            q(0.75),
            q(0.90),
            ts[ts.len() - 1]
        );
        let mean_hit: f64 = res
            .iter()
            .filter(|r| r.hit)
            .map(|r| r.trust as f64)
            .sum::<f64>()
            / res.iter().filter(|r| r.hit).count().max(1) as f64;
        let mean_miss: f64 = res
            .iter()
            .filter(|r| !r.hit)
            .map(|r| r.trust as f64)
            .sum::<f64>()
            / res.iter().filter(|r| !r.hit).count().max(1) as f64;
        println!(
            "mean trust: hits {:.2}  misses {:.2}  (separation {:.2})",
            mean_hit,
            mean_miss,
            mean_hit - mean_miss
        );

        // Trust curve: bin by round(trust), report top-1 hit/family per bin.
        println!("\n  trust  count   cum%   Tool-1%  Fam-1%");
        let maxb = res
            .iter()
            .map(|r| r.trust.round() as i64)
            .max()
            .unwrap_or(0)
            .max(0);
        let mut cum = 0usize;
        for b in 0..=maxb {
            let bin: Vec<&ProjConf> = res.iter().filter(|r| r.trust.round() as i64 == b).collect();
            if bin.is_empty() {
                continue;
            }
            cum += bin.len();
            let h = 100.0 * bin.iter().filter(|r| r.hit).count() as f64 / bin.len() as f64;
            let f = 100.0 * bin.iter().filter(|r| r.fam).count() as f64 / bin.len() as f64;
            println!(
                "  {:>5} {:>6} {:>6.1}  {:>7.1} {:>7.1}",
                b,
                bin.len(),
                100.0 * cum as f64 / n,
                h,
                f
            );
        }

        // Margin curve: how clear-cut top-1 over runner-up predicts correctness.
        println!("\n  margin   count   Tool-1%  Fam-1%");
        for d in 0..10 {
            let lo = d as f32 / 10.0;
            let hi = (d + 1) as f32 / 10.0;
            let bin: Vec<&ProjConf> = res
                .iter()
                .filter(|r| r.margin >= lo && (r.margin < hi || (d == 9 && r.margin <= hi)))
                .collect();
            if bin.is_empty() {
                continue;
            }
            let h = 100.0 * bin.iter().filter(|r| r.hit).count() as f64 / bin.len() as f64;
            let f = 100.0 * bin.iter().filter(|r| r.fam).count() as f64 / bin.len() as f64;
            println!(
                "  {:.1}-{:.1} {:>7}   {:>7.1} {:>7.1}",
                lo,
                hi,
                bin.len(),
                h,
                f
            );
        }

        // Example projections: the turns each links to, with per-link confidence.
        println!("\n  example projections (per-link confidence for the top-5 turns it selects):");
        for r in res.iter().take(6) {
            let flag = if r.hit { "OK " } else { "MISS" };
            println!("   [{}] trust {:.2}  margin {:.2}", flag, r.trust, r.margin);
            for (tool, conv, conf, correct) in &r.links {
                let m = if *correct { "*" } else { " " };
                println!("       {} conf {:>5.2}  conv#{:<6} {}", m, conf, conv, tool);
            }
        }
        return Ok(());
    }

    // ── CONFDIST: per-TOKEN confidence score + how it aligns with correct hit/family ─
    // For each query token, each of the 3 layer-groups picks its best gallery match and
    // casts a z-score-weighted vote; the token's confidence = the z-sum of its winning
    // case (reinforced when groups agree), and its prediction = that case's tool. We bin
    // the confidence and report hit-rate (tool) and family-rate per bin — no threshold,
    // just the score's distribution and its alignment with correctness.
    if std::env::var("CONFDIST").is_ok() {
        let k = 4usize;
        let case_fold: Vec<usize> = cases.iter().map(|c| conv_pos[&c.conv] % k).collect();
        let n_cases = cases.len();
        let mut gal: Vec<(Vec<&WideQSig>, Vec<u32>)> =
            (0..k).map(|_| (Vec::new(), Vec::new())).collect();
        for (ci, c) in cases.iter().enumerate() {
            let cf = case_fold[ci];
            for (f, g) in gal.iter_mut().enumerate() {
                if f != cf {
                    for tok in &c.window {
                        g.0.push(tok);
                        g.1.push(ci as u32);
                    }
                }
            }
        }
        let n_groups = cases
            .iter()
            .flat_map(|c| c.window.first())
            .map(|t| t.n_heads as usize / 4)
            .next()
            .unwrap_or(3);
        let gw = 8usize; // 4 heads × 2 words per folded group

        // (confidence z-sum, correct-tool, correct-family) per query token.
        let recs: Vec<(f32, bool, bool)> = (0..n_cases)
            .into_par_iter()
            .flat_map(|a| {
                let fa = case_fold[a];
                if !(0..n_cases).any(|b| case_fold[b] != fa && cases[b].tool == cases[a].tool) {
                    return Vec::new();
                }
                let (gtok, gcase) = &gal[fa];
                let n_gal = gtok.len().max(1) as f32;
                let tool_a = cases[a].tool.clone();
                let stem_a = stem(&tool_a);
                cases[a]
                    .window
                    .iter()
                    .map(|q| {
                        let mut votes: Vec<(usize, f32)> = Vec::with_capacity(n_groups);
                        for g in 0..n_groups {
                            let base = g * gw;
                            let qg = &q.words[base..base + gw];
                            let (mut best, mut bidx) = (0u32, usize::MAX);
                            let (mut sum, mut sumsq) = (0u64, 0u64);
                            for (j, cand) in gtok.iter().enumerate() {
                                let cw = &cand.words[base..base + gw];
                                let mut ag = 0u32;
                                for kk in 0..8 {
                                    ag += (!(qg[kk] ^ cw[kk])).count_ones();
                                }
                                if ag > best {
                                    best = ag;
                                    bidx = j;
                                }
                                sum += ag as u64;
                                sumsq += (ag as u64) * (ag as u64);
                            }
                            if bidx != usize::MAX {
                                let mean = sum as f32 / n_gal;
                                let var = (sumsq as f32 / n_gal - mean * mean).max(1e-6);
                                let z = ((best as f32 - mean) / var.sqrt()).max(0.0);
                                votes.push((gcase[bidx] as usize, z));
                            }
                        }
                        // Token prediction = case with the highest z-sum across its groups.
                        let (mut best_case, mut best_z) = (usize::MAX, 0f32);
                        for &(c, _) in &votes {
                            let total: f32 = votes
                                .iter()
                                .filter(|&&(cc, _)| cc == c)
                                .map(|&(_, zz)| zz)
                                .sum();
                            if total > best_z {
                                best_z = total;
                                best_case = c;
                            }
                        }
                        let (hit, fam) = if best_case != usize::MAX {
                            (
                                cases[best_case].tool == tool_a,
                                stem(&cases[best_case].tool) == stem_a,
                            )
                        } else {
                            (false, false)
                        };
                        (best_z, hit, fam)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // ── Distribution + alignment ───────────────────────────────────────────
        let n = recs.len().max(1);
        let mut confs: Vec<f32> = recs.iter().map(|r| r.0).collect();
        confs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct_at = |q: f32| confs[((q * (confs.len() - 1) as f32) as usize).min(confs.len() - 1)];
        let mean_conf = confs.iter().sum::<f32>() / n as f32;
        let hit_conf: Vec<f32> = recs.iter().filter(|r| r.1).map(|r| r.0).collect();
        let miss_conf: Vec<f32> = recs.iter().filter(|r| !r.1).map(|r| r.0).collect();
        let mean = |v: &[f32]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        };

        println!("\n══ §79 CONFDIST — per-token confidence (z-sum) vs correctness ══");
        println!(
            "  tokens {n}   overall per-token: hit {:.1}%  fam {:.1}%",
            100.0 * recs.iter().filter(|r| r.1).count() as f64 / n as f64,
            100.0 * recs.iter().filter(|r| r.2).count() as f64 / n as f64
        );
        println!(
            "  confidence  mean {mean_conf:.2}   p10 {:.2}  p50 {:.2}  p90 {:.2}  p99 {:.2}  max {:.2}",
            pct_at(0.10), pct_at(0.50), pct_at(0.90), pct_at(0.99), pct_at(1.0)
        );
        println!(
            "  mean confidence:  hits {:.2}   misses {:.2}   (separation {:.2})",
            mean(&hit_conf),
            mean(&miss_conf),
            mean(&hit_conf) - mean(&miss_conf)
        );

        // Integer-bucketed histogram (bucket = round(conf), capped) — count, hit%, fam%.
        const NB: usize = 24;
        let mut cnt = [0u64; NB];
        let mut hits = [0u64; NB];
        let mut fams = [0u64; NB];
        for &(c, h, f) in &recs {
            let b = (c.round() as usize).min(NB - 1);
            cnt[b] += 1;
            if h {
                hits[b] += 1;
            }
            if f {
                fams[b] += 1;
            }
        }
        let cmax = cnt.iter().copied().max().unwrap_or(1).max(1);
        println!("\n  conf  count    hit%   fam%   distribution");
        for b in 0..NB {
            if cnt[b] == 0 {
                continue;
            }
            let bar = (cnt[b] as usize * 40 / cmax as usize).max(1);
            let lbl = if b == NB - 1 {
                format!("{b}+")
            } else {
                format!("{b}")
            };
            println!(
                "  {lbl:>3}  {:>6}  {:>5.1}  {:>5.1}   {}",
                cnt[b],
                100.0 * hits[b] as f64 / cnt[b] as f64,
                100.0 * fams[b] as f64 / cnt[b] as f64,
                "█".repeat(bar)
            );
        }
        return Ok(());
    }

    // ── LIBCHECK: reproduce the k=4 hit rate using the LIBRARY scorer ───────────
    // Verifies `candle_conversation::provenance::score_provenance_late_fusion` on the
    // pre-folded substrate signatures gives the same numbers as the inline harness.
    if std::env::var("LIBCHECK").is_ok() {
        let k = 4usize;
        let case_fold: Vec<usize> = cases.iter().map(|c| conv_pos[&c.conv] % k).collect();
        let n_cases = cases.len();
        // Per-fold gallery of token refs (tokens whose case is NOT in that fold), so a
        // probe scores only against conversations in other folds — disjoint by conv.
        let mut gal: Vec<(Vec<&WideQSig>, Vec<u32>)> =
            (0..k).map(|_| (Vec::new(), Vec::new())).collect();
        for (ci, c) in cases.iter().enumerate() {
            let cf = case_fold[ci];
            for (f, g) in gal.iter_mut().enumerate() {
                if f != cf {
                    for tok in &c.window {
                        g.0.push(tok);
                        g.1.push(ci as u32);
                    }
                }
            }
        }
        let t = Instant::now();
        let res: Vec<[bool; 4]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                let fa = case_fold[a];
                if cases[a].window.is_empty()
                    || !(0..n_cases).any(|b| case_fold[b] != fa && cases[b].tool == cases[a].tool)
                {
                    return None;
                }
                let (gtok, gcase) = &gal[fa];
                let votes = score_provenance_late_fusion(&cases[a].window, gtok, gcase, n_cases);
                let mut order: Vec<usize> = (0..n_cases).filter(|&b| case_fold[b] != fa).collect();
                order.sort_by(|&x, &y| {
                    votes[y]
                        .partial_cmp(&votes[x])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let st = |b: usize| cases[b].tool == cases[a].tool;
                let sf = |b: usize| stem(&cases[b].tool) == stem(&cases[a].tool);
                Some([
                    st(order[0]),
                    order.iter().take(5).any(|&b| st(b)),
                    sf(order[0]),
                    order.iter().take(5).any(|&b| sf(b)),
                ])
            })
            .collect();
        eprintln!(
            "LIBCHECK scan: {:.2?}  ({} scorable)",
            t.elapsed(),
            res.len()
        );
        println!("\n══ LIBCHECK — library score_provenance_late_fusion, k=4 ══");
        println!(
            "  {:<20} {:>7} {:>7} {:>7} {:>7}",
            "source", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        println!(
            "  {:<20} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            "library z-score LF",
            pct(&res, 0),
            pct(&res, 1),
            pct(&res, 2),
            pct(&res, 3)
        );
        return Ok(());
    }

    // ── RUN1: the locked weight-free config — edges + shift-32 + z-score late fusion ─
    if std::env::var("RUN1").is_ok() {
        let sizes: Vec<usize> = std::env::var("DIST")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![1usize, 2, 20, 2, 20, 2, 1]); // edges default
        let headfold = std::env::var("HEADFOLD").is_ok();
        let hshift: usize = std::env::var("HSHIFT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        // The substrate now stores the pre-folded 12-head (3-group × 4-head) signature —
        // detect it and skip re-folding (raw wide-Q has 192 heads).
        let raw_heads = cases
            .iter()
            .flat_map(|c| c.window.first())
            .map(|t| t.n_heads as usize)
            .next()
            .unwrap_or(0);
        let prefolded = raw_heads == 12;
        let t = Instant::now();
        let f = if prefolded {
            Flat::new(&cases, |w| w.to_vec())
        } else {
            Flat::new(&cases, |w| {
                fold_dist(w, N_LAYERS * HEADS_PER_LAYER, &sizes, 32, headfold, hshift)
            })
        };
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let n_groups = if prefolded { 3 } else { sizes.len() };
        let gw = f.w / n_groups.max(1);
        eprintln!("prefolded={prefolded} raw_heads={raw_heads}  {n_groups} groups × {gw} words");
        let wts = vec![1f32; sizes.len()]; // uniform — no weights; z-score self-weights
        let late = std::env::var("LATE").map(|v| v != "0").unwrap_or(true);
        let conf: u8 = std::env::var("CONF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let flat = std::env::var("FLAT").is_ok();
        eprintln!("mode: flat={flat} late={late} conf={conf}");
        let t = Instant::now();
        let res = if flat {
            run_k(&f, &conv_pos, 4) // pure flat concat: one popcount over all 56 words, one argmax
        } else {
            run_k_weighted(&f, &conv_pos, 4, &wts, gw, late, conf)
        };
        let secs = t.elapsed().as_secs_f64();
        println!(
            "\n══ LOCKED: edges · shift=32 · uniform (weight-free) · z-score · late fusion · k=4 ══"
        );
        println!(
            "  signature   : {} bits/token ({} groups × {} words)",
            f.w * 64,
            n_groups,
            gw
        );
        println!("  gallery     : {} probes, {} tokens", res.len(), f.n_tok);
        println!("  build sig   : {build_ms:.0} ms   scan: {:.2} s", secs);
        println!(
            "  throughput  : {:.0} probes/s   per-probe latency: {:.2} ms  (1 probe = 1 reprojection lookup vs full gallery)",
            res.len() as f64 / secs,
            secs * 1000.0 / res.len().max(1) as f64
        );
        println!(
            "  RESULT      : Tool-1 {:.1}   Tool-5 {:.1}   Fam-1 {:.1}   Fam-5 {:.1}",
            pct(&res, 0),
            pct(&res, 1),
            pct(&res, 2),
            pct(&res, 3)
        );
        return Ok(());
    }

    // ── GW: weighted layer-group voting over the `edges` distribution ───────────
    if let Ok(gw_mode) = std::env::var("GW") {
        let sizes = vec![1usize, 2, 20, 2, 20, 2, 1]; // edges
        let sh = if shift > 0 { shift } else { 32 }; // edges settled with the 32-bit stagger
        let f = Flat::new(&cases, |w| {
            fold_dist(w, N_LAYERS * HEADS_PER_LAYER, &sizes, sh, false, 0)
        });
        let n_groups = sizes.len();
        let gw = f.w / n_groups; // words per group
        let mut labels = Vec::new();
        let mut l0 = 0usize;
        for &sz in &sizes {
            labels.push(if sz == 1 {
                format!("L{l0}")
            } else {
                format!("L{}-{}", l0, l0 + sz - 1)
            });
            l0 += sz;
        }
        let late = std::env::var("LATE").is_ok();
        let conf: u8 = std::env::var("CONF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let row = |name: &str, wts: &[f32], conv_pos: &HashMap<u64, usize>| {
            let res = run_k_weighted(&f, conv_pos, 4, wts, gw, late, conf);
            let ws: Vec<String> = wts.iter().map(|w| format!("{w:.2}")).collect();
            println!(
                "  {name:<10} {:>7.1} {:>7.1} {:>7.1} {:>7.1}   [{}]",
                pct(&res, 0),
                pct(&res, 1),
                pct(&res, 2),
                pct(&res, 3),
                ws.join(", ")
            );
        };
        let fusion = if late {
            "LATE (per-group vote tally)"
        } else {
            "early (combine then argmax)"
        };
        println!("\n══ edges (shift={sh}) group voting [{fusion}] — rolling-{back} k=4 ══");
        println!(
            "  {:<10} {:>7} {:>7} {:>7} {:>7}",
            "config", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        // Per-group solo (unit weight) — how discriminative each group is alone.
        println!("  -- per-group solo --");
        let mut solo = vec![0f32; n_groups];
        for g in 0..n_groups {
            let mut wts = vec![0f32; n_groups];
            wts[g] = 1.0;
            let res = run_k_weighted(&f, &conv_pos, 4, &wts, gw, late, conf);
            solo[g] = pct(&res, 0) as f32;
            println!(
                "  g{g} {:<7} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                labels[g],
                pct(&res, 0),
                pct(&res, 1),
                pct(&res, 2),
                pct(&res, 3)
            );
        }
        println!("  -- weightings --");
        row("uniform", &vec![1f32; n_groups], &conv_pos);
        row("linear", &solo, &conv_pos);
        let sq: Vec<f32> = solo.iter().map(|x| x * x).collect();
        row("squared", &sq, &conv_pos);
        let mn = solo.iter().cloned().fold(f32::MAX, f32::min);
        let rel: Vec<f32> = solo.iter().map(|x| (x - mn).max(0.0)).collect();
        row("above-min", &rel, &conv_pos);
        let cube: Vec<f32> = rel.iter().map(|x| x * x * x).collect();
        row("relcube", &cube, &conv_pos);
        // Binary keep/drop: keep only groups whose solo Tool-1 clears a threshold.
        let keep = |thr: f32| -> Vec<f32> {
            solo.iter()
                .map(|&x| if x >= thr { 1.0 } else { 0.0 })
                .collect()
        };
        row("keep>60", &keep(60.0), &conv_pos); // g1,g3,g5,g6
        row("keep>72", &keep(72.0), &conv_pos); // g3,g5,g6
                                                // Weight the kept groups by their solo strength (best of both).
        let kw: Vec<f32> = solo
            .iter()
            .map(|&x| if x >= 60.0 { x } else { 0.0 })
            .collect();
        row("keep>60·lin", &kw, &conv_pos);
        // Custom weight list "1,0,0,1,0,3,2".
        if gw_mode != "all" {
            let custom: Vec<f32> = gw_mode
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            if custom.len() == n_groups {
                row("custom", &custom, &conv_pos);
            } else {
                eprintln!("GW custom needs {n_groups} weights, got {}", custom.len());
            }
        }
        return Ok(());
    }

    // ── DIST: variable-size layer-fold groups (bottom→top), with the 32-bit shift ─
    if let Ok(want) = std::env::var("DIST") {
        let sets: Vec<(String, Vec<usize>)> = if want == "all" {
            dist_set()
                .into_iter()
                .map(|(n, s)| (n.to_string(), s))
                .collect()
        } else {
            vec![(
                "custom".to_string(),
                want.split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect(),
            )]
        };
        println!("\n══ Layer-fold distributions — rolling-{back} k=4, shift={shift} ══");
        println!(
            "  {:<10} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7}   sizes (bottom→top)",
            "dist", "grps", "bits", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        for (name, sizes) in sets {
            let covered: usize = sizes.iter().sum::<usize>().min(N_LAYERS);
            let f = Flat::new(&cases, |w| {
                fold_dist(w, N_LAYERS * HEADS_PER_LAYER, &sizes, shift, false, 0)
            });
            let res = run_k(&f, &conv_pos, 4);
            let mark = if covered != N_LAYERS {
                format!(" [covers {covered}/48]")
            } else {
                String::new()
            };
            println!(
                "  {:<10} {:>6} {:>6} {:>7.1} {:>7.1} {:>7.1} {:>7.1}   {:?}{}",
                name,
                sizes.len(),
                f.w * 64,
                pct(&res, 0),
                pct(&res, 1),
                pct(&res, 2),
                pct(&res, 3),
                sizes,
                mark
            );
        }
        return Ok(());
    }

    // ── LSET: use only a discrete set of layers (full 128 bits each) ────────────
    if let Ok(s) = std::env::var("LSET") {
        let layers: Vec<usize> = s
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .filter(|&l| l < N_LAYERS)
            .collect();
        let mut budget = [0usize; 48];
        for &l in &layers {
            budget[l] = 128;
        }
        let f = Flat::new(&cases, |w| build_sig(w, &budget));
        let t = Instant::now();
        let res = run_k(&f, &conv_pos, 4);
        println!(
            "LSET={:<20} bits={:>5} scan={:>5.2}s  Tool-1 {:.1}  Tool-5 {:.1}  Fam-1 {:.1}  Fam-5 {:.1}",
            format!("{layers:?}"),
            f.w * 64,
            t.elapsed().as_secs_f64(),
            pct(&res, 0),
            pct(&res, 1),
            pct(&res, 2),
            pct(&res, 3)
        );
        return Ok(());
    }

    // ── Asymmetric bit-budget schemes: one signature layout per named scheme ────
    if let Ok(want) = std::env::var("SCHEME") {
        println!("\n══ Asymmetric bit-budget — rolling-{back} k=4 (disjoint by conversation) ══");
        println!(
            "  {:<12} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "scheme", "bits", "scan_s", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        for (name, budget) in scheme_budgets() {
            if want != "all" && want != name {
                continue;
            }
            let f = Flat::new(&cases, |w| build_sig(w, &budget));
            let t = Instant::now();
            let res = run_k(&f, &conv_pos, 4);
            println!(
                "  {:<12} {:>8} {:>7.2} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                name,
                f.w * 64,
                t.elapsed().as_secs_f64(),
                pct(&res, 0),
                pct(&res, 1),
                pct(&res, 2),
                pct(&res, 3)
            );
        }
        return Ok(());
    }

    // ── LFOLD sweep: fold + scan (k=4) at each layer-group size, single load ────
    if let Some(sweep) = std::env::var("LFOLD_SWEEP").ok().map(|s| parse_sweep(&s)) {
        println!(
            "\n══ LFOLD sweep — rolling-{back} k=4 (probe/gallery disjoint by conversation) ══"
        );
        println!(
            "  {:>5} {:>7} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "LFOLD", "groups", "bits", "scan_s", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        for lf in sweep {
            let f = Flat::build(&cases, lf, lsel, shift);
            let t = Instant::now();
            let res = run_k(&f, &conv_pos, 4);
            let secs = t.elapsed().as_secs_f64();
            println!(
                "  {:>5} {:>7} {:>8} {:>7.2} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                lf,
                48usize.div_ceil(lf.max(1)),
                f.w * 64,
                secs,
                pct(&res, 0),
                pct(&res, 1),
                pct(&res, 2),
                pct(&res, 3)
            );
        }
        return Ok(());
    }

    let f = Flat::build(&cases, lfold, lsel, shift);
    eprintln!(
        "layer fold          : LFOLD={lfold}  → 48 layers collapsed to {} groups ({} bits/tok)",
        48usize.div_ceil(lfold.max(1)),
        f.w * 64
    );

    // ── §77 — leave-one-conversation-out (opt-in: it's the k→∞ reference) ───────
    if std::env::var("LOO").is_ok() {
        let t = Instant::now();
        let res77 = (0..f.n_cases)
            .into_par_iter()
            .filter_map(|a| score_case(&f, a, &f.tok_gconv, &f.case_gconv, f.case_gconv[a]))
            .collect::<Vec<_>>();
        eprintln!("§77 scan            : {:.2?}", t.elapsed());
        println!(
            "\n══ §77 — rolling-{back} LOO ({} scorable) ══",
            res77.len()
        );
        println!(
            "  {:<16} {:>7} {:>7} {:>7} {:>7}",
            "source", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        println!(
            "  {:<16} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            "aligned wide-Q",
            pct(&res77, 0),
            pct(&res77, 1),
            pct(&res77, 2),
            pct(&res77, 3)
        );
    }

    // ── §78 — k-fold, probe/gallery disjoint by conversation ───────────────────
    let ks: Vec<usize> = std::env::var("S78_KS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4]);
    println!("\n══ §78 — rolling-{back} k-fold (probe/gallery disjoint by conversation) ══");
    println!(
        "  {:<10} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "config", "n", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
    );
    for &k in &ks {
        let t = Instant::now();
        let res = run_k(&f, &conv_pos, k);
        eprintln!("§78 k={k} scan        : {:.2?}", t.elapsed());
        println!(
            "  {:<10} {:>7} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            format!("k={k}"),
            res.len(),
            pct(&res, 0),
            pct(&res, 1),
            pct(&res, 2),
            pct(&res, 3)
        );
    }
    Ok(())
}
