//! `calibrate_alignment` — reproduce the §16 call→definition CCA (Top-1 90% /
//! Top-5 95%) in a standalone harness, then generalize it to corpus-trained
//! calibration while holding 90%.
//!
//! Step 1 (this build): the **control** — fit the CCA directly on call↔def pairs
//! with §16's exact recipe and confirm ~90% on held-out tool calls. Only once the
//! pipeline reproduces the baseline is any corpus comparison meaningful.
//!
//! §16 recipe (locked): K over **L24–40**, probe = first 8 tokens of the call
//! (per-token, raw), def = first 12 tokens (mean-pooled, raw), per-token×per-copy
//! pairs, PCA→d, CCA→r=32, mean-pool the probe at eval, fold-by-tool max.
//!
//! ```bash
//! cargo run -p zend --example calibrate_alignment --release -- .
//! ```

// Offline calibration harness. The numerical passes walk layer × head × dim
// index grids to mirror the shapes in the write-up, and several report structs
// exist only so a stage can be re-run in isolation — so the index loops and the
// unconstructed variants are deliberate here, not leftovers.
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_sort_by,
    clippy::useless_vec,
    clippy::for_kv_map,
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::doc_lazy_continuation,
    clippy::print_literal,
    clippy::repeat_vec_with_capacity,
    dead_code
)]

use std::collections::HashMap;
use std::path::PathBuf;

use candle::quantized::ggml_file::qtensor_from_ggml;
use candle::quantized::k_quants::{BlockQ4_KS, BlockQ8_KS, GgmlType};
use candle::quantized::GgmlDType;
use candle::{DType, Device};
use candle_conversation::persistence::record::ChunkPayload;
use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{StreamDecl, StreamId};
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;
use candle_nn::kv_cache::KvFormat;
use tokenizers::Tokenizer;

const N_LAYERS: usize = 48;
const N_KV_HEAD: usize = 4;
const HEAD_DIM: usize = 128;
const N_PAL: usize = 4;
const SUB: usize = HEAD_DIM / N_PAL; // 32
const ELEMS_PER_SUBBAND: usize = 32 * SUB; // 1024
const FLOATS_PER_HEAD: usize = N_PAL * ELEMS_PER_SUBBAND; // 4096
const PER_LAYER_DIM: usize = N_KV_HEAD * HEAD_DIM; // 512

const BAND_LO: usize = 0;
const BAND_HI: usize = 48; // FULL stack (all 48 layers). (§50's noise floor needs
                           // L24..40 to match the 8192-dim calibration — set 24/40 to
                           // run §50; full stack here for §51's blind-selection suite.)
const PWIN: usize = 12; // probe call window (sweet spot)
const SWIN: usize = 32; // def window (full first block)

// ── KV dequant (the proven research-harness path) ──────────────────────────────
fn kv_ggml(fmt: KvFormat) -> (GgmlDType, usize) {
    match fmt {
        KvFormat::Quantized(qf) => (
            (qf.to_ggml_dtype()),
            (ELEMS_PER_SUBBAND / qf.block_size()) * qf.bytes_per_block(),
        ),
        KvFormat::Float(dt) => {
            let (g, bpe) = match dt {
                DType::F32 => (GgmlDType::F32, 4),
                DType::BF16 => (GgmlDType::BF16, 2),
                _ => (GgmlDType::F16, 2),
            };
            (g, ELEMS_PER_SUBBAND * bpe)
        }
    }
}

fn dequant_subband_k(bytes: &[u8], fmt: KvFormat) -> Option<Vec<f32>> {
    let (g, bsize) = kv_ggml(fmt);
    if g == GgmlDType::R16 {
        let mut out = vec![0.0f32; ELEMS_PER_SUBBAND];
        for d in 0..SUB {
            for t in 0..32 {
                let off = d * 128 + t * 2;
                out[t * SUB + d] = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
            }
        }
        return Some(out);
    }
    if bytes.len() < bsize {
        return None;
    }
    let n_blocks = ELEMS_PER_SUBBAND / 32;
    let mut out = vec![0.0f32; ELEMS_PER_SUBBAND];
    match fmt.as_quant() {
        Some(candle_nn::kv_cache::QuantFormat::Q8_KS) => {
            let blocks = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ8_KS, n_blocks)
            };
            BlockQ8_KS::to_float(blocks, &mut out);
            Some(out)
        }
        Some(candle_nn::kv_cache::QuantFormat::Q4_KS) => {
            let blocks = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4_KS, n_blocks)
            };
            BlockQ4_KS::to_float(blocks, &mut out);
            Some(out)
        }
        _ => {
            let qt = qtensor_from_ggml(g, &bytes[..bsize], vec![ELEMS_PER_SUBBAND], &Device::Cpu)
                .ok()?;
            qt.dequantize(&Device::Cpu)
                .ok()?
                .flatten_all()
                .ok()?
                .to_vec1::<f32>()
                .ok()
        }
    }
}

fn extract_k_block(payload: &ChunkPayload) -> Option<Vec<f32>> {
    let mut off = 0usize;
    let mut k = vec![0.0f32; N_KV_HEAD * FLOATS_PER_HEAD];
    for h in 0..N_KV_HEAD {
        for p in 0..N_PAL {
            let kfmt = KvFormat::from_tag(*payload.k_formats.get(h * N_PAL + p)?)?;
            let (_, kbytes) = kv_ggml(kfmt);
            if off + kbytes > payload.kv_bytes.len() {
                return None;
            }
            let kf = dequant_subband_k(&payload.kv_bytes[off..off + kbytes], kfmt)?;
            for t in 0..32 {
                for d in 0..SUB {
                    k[h * FLOATS_PER_HEAD + p * ELEMS_PER_SUBBAND + t * SUB + d] = kf[t * SUB + d];
                }
            }
            off += kbytes;
            let vfmt = KvFormat::from_tag(*payload.v_formats.get(h * N_PAL + p)?)?;
            off += kv_ggml(vfmt).1;
        }
    }
    Some(k)
}

fn token_k(kblock: &[f32], t_in: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; PER_LAYER_DIM];
    for h in 0..N_KV_HEAD {
        for p in 0..N_PAL {
            for dd in 0..SUB {
                out[h * HEAD_DIM + p * SUB + dd] =
                    kblock[h * FLOATS_PER_HEAD + p * ELEMS_PER_SUBBAND + t_in * SUB + dd];
            }
        }
    }
    out
}

/// Per-token band vectors over the routing band (L24–40): `[token] = 8192-dim`.
fn read_token_bands(
    persistence: &mut SubstratePersistence,
    substrate: &Substrate,
    stream_id: StreamId,
    n_tok: usize,
) -> Option<Vec<Vec<f32>>> {
    let chunks = persistence.read_stream_chunks(substrate, stream_id).ok()?;
    let n_chunks = chunks.len();
    if n_chunks == 0 || n_chunks % N_LAYERS != 0 {
        return None;
    }
    let cpl = n_chunks / N_LAYERS;
    let mut by_idx: HashMap<u64, &ChunkPayload> = HashMap::with_capacity(n_chunks);
    for (ci, p) in &chunks {
        by_idx.insert(*ci, p);
    }
    let band = BAND_HI - BAND_LO;
    let mut out = vec![vec![0.0f32; band * PER_LAYER_DIM]; n_tok];
    for (lpos, l) in (BAND_LO..BAND_HI).enumerate() {
        for block in 0..cpl {
            let payload = by_idx.get(&((l * cpl + block) as u64))?;
            let kb = extract_k_block(payload)?;
            for t_in in 0..32 {
                let gt = block * 32 + t_in;
                if gt >= n_tok {
                    break;
                }
                let kv = token_k(&kb, t_in);
                out[gt][lpos * PER_LAYER_DIM..(lpos + 1) * PER_LAYER_DIM].copy_from_slice(&kv);
            }
        }
    }
    Some(out)
}

// ── R16 Q/K extraction (quantization disabled → lossless probe blocks) ───────
// The R16 KV block is { f16 K[32] , f16 Q[32] } per (head,palette,dim), so a
// probe's chunks carry BOTH the key and the query. Mirrors
// `tool_provenance_research::parse_r16_chunk`.
const K_SUBBAND_BYTES: usize = SUB * 128; // 4096 (32 dims × {K[32]+Q[32]} f16)
const V_SUBBAND_F16_BYTES: usize = 32 * SUB * 2; // 2048 (probe V is F16)

/// Parse one R16 chunk's `kv_bytes` into `(q, k)` flat arrays, each in the same
/// `[head][palette][token][sub_dim]` order `token_k` reshapes. `None` if the blob
/// is not a full R16-K + F16-V chunk (e.g. a quantised def, not a probe).
fn parse_r16_chunk(kv: &[u8]) -> Option<(Vec<f32>, Vec<f32>)> {
    let stride_hp = K_SUBBAND_BYTES + V_SUBBAND_F16_BYTES; // 6144
    if kv.len() < N_KV_HEAD * N_PAL * stride_hp {
        return None;
    }
    let f16 = |b: &[u8], o: usize| half::f16::from_le_bytes([b[o], b[o + 1]]).to_f32();
    let mut q = vec![0.0f32; N_KV_HEAD * FLOATS_PER_HEAD];
    let mut k = vec![0.0f32; N_KV_HEAD * FLOATS_PER_HEAD];
    for h in 0..N_KV_HEAD {
        for p in 0..N_PAL {
            let kbase = (h * N_PAL + p) * stride_hp;
            for d in 0..SUB {
                let blk = kbase + d * 128;
                for t in 0..32 {
                    let idx = h * FLOATS_PER_HEAD + p * ELEMS_PER_SUBBAND + t * SUB + d;
                    k[idx] = f16(kv, blk + t * 2);
                    q[idx] = f16(kv, blk + 64 + t * 2);
                }
            }
        }
    }
    Some((q, k))
}

/// Per-token band vectors for BOTH query and key over the routing band (L24–40):
/// `(q_bands, k_bands)`, each `[token] = 8192-d`. Probe (R16) streams only.
fn read_token_qk_bands(
    persistence: &mut SubstratePersistence,
    substrate: &Substrate,
    stream_id: StreamId,
    n_tok: usize,
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let chunks = persistence.read_stream_chunks(substrate, stream_id).ok()?;
    let n_chunks = chunks.len();
    if n_chunks == 0 || n_chunks % N_LAYERS != 0 {
        return None;
    }
    let cpl = n_chunks / N_LAYERS;
    let mut by_idx: HashMap<u64, &ChunkPayload> = HashMap::with_capacity(n_chunks);
    for (ci, p) in &chunks {
        by_idx.insert(*ci, p);
    }
    let band = BAND_HI - BAND_LO;
    let mut q_out = vec![vec![0.0f32; band * PER_LAYER_DIM]; n_tok];
    let mut k_out = vec![vec![0.0f32; band * PER_LAYER_DIM]; n_tok];
    for (lpos, l) in (BAND_LO..BAND_HI).enumerate() {
        for block in 0..cpl {
            let payload = by_idx.get(&((l * cpl + block) as u64))?;
            let (qb, kb) = parse_r16_chunk(&payload.kv_bytes)?;
            for t_in in 0..32 {
                let gt = block * 32 + t_in;
                if gt >= n_tok {
                    break;
                }
                let qv = token_k(&qb, t_in);
                let kv = token_k(&kb, t_in);
                q_out[gt][lpos * PER_LAYER_DIM..(lpos + 1) * PER_LAYER_DIM].copy_from_slice(&qv);
                k_out[gt][lpos * PER_LAYER_DIM..(lpos + 1) * PER_LAYER_DIM].copy_from_slice(&kv);
            }
        }
    }
    Some((q_out, k_out))
}

/// Z-normalise a score vector → `(x − mean)/std`, so two channels can be blended
/// on a common scale (the §7 "per-query z-normalise each" fusion recipe).
fn znorm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    if n == 0.0 {
        return Vec::new();
    }
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std = var.sqrt().max(1e-6);
    x.iter().map(|v| (v - mean) / std).collect()
}

fn mean_band(bands: &[Vec<f32>], range: std::ops::Range<usize>) -> Vec<f32> {
    let dim = bands.first().map(|b| b.len()).unwrap_or(0);
    let mut acc = vec![0.0f32; dim];
    let mut n = 0f32;
    for t in range {
        if t < bands.len() {
            for (a, x) in acc.iter_mut().zip(&bands[t]) {
                *a += x;
            }
            n += 1.0;
        }
    }
    if n > 0.0 {
        for x in acc.iter_mut() {
            *x /= n;
        }
    }
    acc
}

// ── ChatML token structure ──────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum Role {
    System,
    User,
    Assistant,
    Other,
}
struct Msg {
    role: Role,
    start: usize,
    end: usize,
}
struct Markers {
    im_start: u32,
    im_end: u32,
    think_end: Option<u32>,
}
fn detok(tok: &Tokenizer, id: u32) -> String {
    tok.id_to_token(id)
        .unwrap_or_default()
        .replace('\u{0120}', " ")
        .replace('\u{010A}', "\n")
}
/// Structural token: carries no tool identity, only scaffold/formatting. Dropped from
/// BOTH the probe decode tokens and the def tokens we score against (it is shared
/// common-mode noise). Covers four classes:
///   1. whitespace-only — spaces, tabs, newlines (`Ġ`/`Ċ` decode to ` `/`\n`);
///   2. pure punctuation / JSON syntax — `{ } [ ] " ' : ,` and friends;
///   3. special / markup tokens emitted whole — `<tool_call>`, `</think>`,
///      `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, …;
///   4. tool-call JSON scaffold keywords — `name`, `arguments`, `tool_call`, …
/// `KEEP_STRUCT=1` disables structural filtering everywhere (all tokens kept on both
/// the decode and definition sides) — the experiment toggle. Cached once.
fn keep_struct() -> bool {
    use std::sync::OnceLock;
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| std::env::var("KEEP_STRUCT").is_ok())
}
/// The structural rules themselves, ALWAYS applied (ignores `KEEP_STRUCT`). Used to
/// build a content-only def mean K even when scoring keeps all tokens (§49 decouples
/// the similarity from the scoring).
fn is_structural_rules(tok: &Tokenizer, id: u32) -> bool {
    let s = detok(tok, id);
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    if t.chars()
        .all(|c| "{}[]\"':,<>|`.!?;-_=/\\()~@#$%^&*+".contains(c))
    {
        return true;
    }
    if t.starts_with('<') && t.ends_with('>') {
        return true;
    }
    matches!(
        t.to_ascii_lowercase().as_str(),
        "name" | "arguments" | "tool_call" | "tool_response" | "think" | "function" | "parameters"
    )
}
fn is_structural(tok: &Tokenizer, id: u32) -> bool {
    if keep_struct() {
        return false;
    }
    is_structural_rules(tok, id)
}
/// A sub-selection of a band vector — one routing-band layer or one KV head.
#[derive(Clone, Copy)]
enum Sel {
    Layer(usize),
    Head(usize),
}

/// Mean-pool the bands of `idxs` (token indices) → one band vector.
fn mean_of(bands: &[Vec<f32>], idxs: &[usize]) -> Vec<f32> {
    let dim = bands.first().map(|b| b.len()).unwrap_or(0);
    let mut acc = vec![0.0f32; dim];
    for &t in idxs {
        for (a, x) in acc.iter_mut().zip(&bands[t]) {
            *a += x;
        }
    }
    let n = idxs.len().max(1) as f32;
    for x in acc.iter_mut() {
        *x /= n;
    }
    acc
}
fn parse_messages(toks: &[u32], m: &Markers, tok: &Tokenizer) -> Vec<Msg> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] != m.im_start {
            i += 1;
            continue;
        }
        let role = if i + 1 < toks.len() {
            match detok(tok, toks[i + 1]).trim() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => Role::Other,
            }
        } else {
            Role::Other
        };
        let mut j = i + 2;
        while j < toks.len() && toks[j] != m.im_end {
            j += 1;
        }
        out.push(Msg {
            role,
            start: (i + 2).min(toks.len()),
            end: j.min(toks.len()),
        });
        i = j + 1;
    }
    out
}
fn tool_name_from_text(s: &str) -> Option<String> {
    let i = s.find("\"name\"")?;
    let after = &s[i + 6..];
    let colon = after.find(':')?;
    let rest = &after[colon + 1..];
    let q1 = rest.find('"')?;
    let q2 = rest[q1 + 1..].find('"')?;
    let name = &rest[q1 + 1..q1 + 1 + q2];
    (!name.is_empty()).then(|| name.to_string())
}

#[derive(Default, Clone, Copy)]
struct Acc {
    n: usize,
    t1: usize,
    t5: usize,
}
impl Acc {
    fn add(&mut self, rank: Option<usize>) {
        self.n += 1;
        if let Some(r) = rank {
            if r == 0 {
                self.t1 += 1;
            }
            if r < 5 {
                self.t5 += 1;
            }
        }
    }
}

/// Rank `tool` in a per-tool score map (descending score).
fn rank_tool(per: &HashMap<&str, f32>, tool: &str) -> Option<usize> {
    let mut v: Vec<(&str, f32)> = per.iter().map(|(k, vv)| (*k, *vv)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.iter().position(|(t, _)| *t == tool)
}

/// Pack a band's per-dimension signs into u64 words (bit i set ⇔ dim i ≥ 0) for
/// XOR+popcount Hamming — the RAW BDP signature, `sign(K)`, no rotation.
fn sign_pack(b: &[f32]) -> Vec<u64> {
    let mut w = vec![0u64; b.len().div_ceil(64)];
    for (i, &x) in b.iter().enumerate() {
        if x >= 0.0 {
            w[i / 64] |= 1u64 << (i % 64);
        }
    }
    w
}
/// One token's matching sign bits vs a def over `dims` dimensions (`dims −
/// Hamming`) — the per-token XOR-popcount the §21.2 readout sums across tokens.
fn sign_match(a: &[u64], b: &[u64], dims: usize) -> f32 {
    let diff: u32 = a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
    (dims as u32 - diff) as f32
}
/// Per-head XOR popcount (Hamming) from a flat band-order signature. The band is
/// `[layer][head(128)]`, and `l·512 + h·128` is always 64-aligned, so head `h`'s
/// bits are the u64 words `{l·8 + h·2, +1}` per layer (PER_LAYER_DIM=512=8 words,
/// HEAD_DIM=128=2 words). Used by §23's per-head distribution split.
fn xor_pop_head(a: &[u64], b: &[u64], h: usize) -> u32 {
    const LW: usize = PER_LAYER_DIM / 64; // 8 words per layer
    const HW: usize = HEAD_DIM / 64; // 2 words per head
    let nlayers = a.len() / LW;
    let mut diff = 0u32;
    for l in 0..nlayers {
        let base = l * LW + h * HW;
        for w in 0..HW {
            diff += (a[base + w] ^ b[base + w]).count_ones();
        }
    }
    diff
}
/// (mean, population std) of a slice.
fn mean_std(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let m = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
    (m, var.sqrt())
}
/// Cohen's d: `(mean_a − mean_b) / pooled_std`. `+` ⇒ `a` (correct) sits higher.
fn cohens_d(a: &[f32], b: &[f32]) -> f32 {
    let (ma, sa) = mean_std(a);
    let (mb, sb) = mean_std(b);
    let pooled = (0.5 * (sa * sa + sb * sb)).sqrt().max(1e-9);
    (ma - mb) / pooled
}
/// `q`-th percentile (`q ∈ [0,1]`) of an already-sorted slice.
fn pct_of(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((q * (sorted.len() - 1) as f32).round() as usize).min(sorted.len() - 1);
    sorted[i]
}

/// A turn's complete per-token wide `sign(Q)` history from its `WideQSig` record —
/// the widened `Signatures` corpus the §74/§77/§78 retrieval scans.
fn final_wide_window(bytes: &[u8]) -> Option<Vec<candle_conversation::provenance::WideQSig>> {
    candle_conversation::provenance::decode_wide_sigs(bytes)
}

/// Rolling wide-Q lookback: how many tokens of live `sign(Q)` immediately
/// preceding a projection point form that projection's lookup signature. This is
/// the production reprojection query — at each reprojection the retriever keys on
/// the last `ROLLING_BACK` tokens of Q, not the whole turn.
const ROLLING_BACK: usize = 64;

/// One retrieval probe = a single projection event: the rolling wide-Q window
/// ending at the projection point, the `tools` section that projection locked,
/// and the conversation (turn stream) it belongs to so probe and gallery can be
/// held out by conversation.
struct ProbeCase {
    tool: String,
    conv: u64,
    window: Vec<candle_conversation::provenance::WideQSig>,
}

/// Build per-projection probe cases from the substrate. For every projection
/// event that locked a `tools` section, the lookup signature is the last
/// `ROLLING_BACK` tokens of the turn's wide-Q `sign(Q)` history ending at the
/// projection point (`assistant_content_start + end_token`). Early projections
/// naturally reach back through the think block into the user-prompt prefill;
/// later ones ride the reasoning — exactly the query production reprojection
/// issues. The tool label comes from the projection event's own selection, so
/// each probe's ground truth is the substrate-native provenance marker.
fn projection_probe_cases(substrate: &Substrate) -> Vec<ProbeCase> {
    use candle_conversation::projection::{decode_events, SystemItem};

    let mut cases = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(d)) = &e.decl else {
            continue;
        };
        let Some(history) = e.wide_q_sigs.as_ref().and_then(|b| final_wide_window(b)) else {
            continue;
        };
        if history.is_empty() {
            continue;
        }
        let Some(events) = e.projection_events.as_ref().map(|b| decode_events(b)) else {
            continue;
        };
        let asst = d.assistant_content_start() as usize;
        for ev in &events {
            // The `tools` section this projection locked — the provenance label.
            let Some(tool) = ev.selection.system.iter().find_map(|item| match item {
                SystemItem::Collection { name, sections } if name == "tools" => {
                    sections.iter().find(|s| s.selected).map(|s| s.name.clone())
                }
                _ => None,
            }) else {
                continue; // this projection didn't lock a tool — not a scorable probe
            };
            // Rolling-`ROLLING_BACK` wide-Q window ending at the projection point
            // (point-model event: `start_token` is the generated position at
            // which the projection was applied).
            let point = (asst + ev.start_token as usize).min(history.len());
            let lo = point.saturating_sub(ROLLING_BACK);
            let window = history[lo..point].to_vec();
            if window.is_empty() {
                continue;
            }
            cases.push(ProbeCase {
                tool,
                conv: sid.0,
                window,
            });
        }
    }
    cases
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    eprintln!("loading substrate at {} …", workspace.display());
    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open: {e}"))?;
    let tok = Tokenizer::from_file(workspace.join(".substrate").join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tok: {e}"))?;

    // ── KCHECK: for a turn with dead wide-Q tokens, are the K values also zero? ──
    //  Decode's `write_regs_to_r16` writes K (d[]) and Q (q[]) in one loop, so if
    //  the write never ran BOTH are zero → attention degraded. If K is present
    //  while q[] is dead, the loss is Q-specific (a post-write snapshot path that
    //  carries d[] but drops q[]) and attention is unaffected. This reads BOTH
    //  from the persisted R16 chunk and reports per-token ||K|| vs ||Q||.
    if std::env::var("KCHECK").is_ok() {
        use candle_conversation::persistence::resume::decode_token_ids;
        use candle_conversation::provenance::decode_wide_sigs;
        let want = std::env::args().nth(2).unwrap_or_default();
        let norm = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
        let zeros = |v: &[f32]| -> usize { v.iter().filter(|&&x| x == 0.0).count() };
        for (sid, e) in substrate.all_streams() {
            if !matches!(e.decl, Some(StreamDecl::Turn(_))) {
                continue;
            }
            let Some(hist) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
                continue;
            };
            let Some(ids) = persistence
                .read_tokens(&substrate, sid)
                .ok()
                .flatten()
                .and_then(|b| decode_token_ids(&b).ok())
            else {
                continue;
            };
            let full = hist.first().map(|s| s.n_heads as u32 * 128).unwrap_or(0);
            let dead: Vec<usize> = hist
                .iter()
                .enumerate()
                .filter(|(_, s)| s.popcount() == full || s.popcount() == 0)
                .map(|(i, _)| i)
                .collect();
            if dead.is_empty() {
                continue;
            }
            let text = tok.decode(&ids, false).unwrap_or_default();
            if !want.is_empty() && !text.contains(&want) {
                continue;
            }
            let n_tok = ids.len();
            let Some((q_bands, k_bands)) =
                read_token_qk_bands(&mut persistence, &substrate, sid, n_tok)
            else {
                eprintln!("stream {}: no R16 chunk read", sid.0);
                continue;
            };
            let dim = q_bands.first().map(|v| v.len()).unwrap_or(0);
            println!(
                "\n=== KCHECK stream {} ({n_tok} Tokens, {} wide-Q, {} dead, band dim {dim}) ===",
                sid.0,
                hist.len(),
                dead.len()
            );
            println!(
                "  per-token: ||K|| = key norm, ||Q|| = query norm, z = zero-count of that band"
            );
            let mut k_alive_on_dead = 0usize;
            let mut checked = 0usize;
            for &i in &dead {
                if i >= n_tok {
                    continue; // trailing over-capture: no persisted chunk token
                }
                checked += 1;
                let kn = norm(&k_bands[i]);
                let qn = norm(&q_bands[i]);
                if kn > 1e-6 {
                    k_alive_on_dead += 1;
                }
                if checked <= 12 {
                    println!(
                        "  idx {i:>4} {:>14}  ||K||={kn:>9.3} (z {:>5}/{dim})   ||Q||={qn:>9.3} (z {:>5}/{dim})",
                        format!("{:?}", tok.id_to_token(ids[i]).unwrap_or_default()),
                        zeros(&k_bands[i]),
                        zeros(&q_bands[i]),
                    );
                }
            }
            // A live baseline: a mid-think token that is NOT dead.
            if let Some(&alive) = (4..n_tok).find(|j| !dead.contains(j)).as_ref() {
                println!(
                    "  --- live baseline idx {alive}: ||K||={:>9.3} ||Q||={:>9.3}",
                    norm(&k_bands[alive]),
                    norm(&q_bands[alive])
                );
            }
            println!(
                "  VERDICT: of {checked} in-range dead-Q tokens, {k_alive_on_dead} have LIVE K (||K||>0).  \
                 {} → K present, Q-only loss (attention OK)",
                if k_alive_on_dead == checked { "all" } else { "NOT all" }
            );
            return Ok(());
        }
        println!("(no dead-Q turn matched)");
        return Ok(());
    }

    // ── CHUNKLAYOUT: per-chunk K-fill map — is the "hole" partial-chunk padding? ──
    //  Dumps layer-0 chunks in order with their actual filled-slot pattern. If
    //  interior chunks are partial (X…X..… trailing dots) with the NEXT chunk
    //  continuing the sequence, the wide-Q "dead tokens" are legitimate padding that
    //  the block×32 gather mislabels — NOT attention-visible corruption.
    if std::env::var("CHUNKLAYOUT").is_ok() {
        use candle_conversation::persistence::resume::decode_token_ids;
        use candle_conversation::provenance::decode_wide_sigs;
        let want = std::env::args().nth(2).unwrap_or_default();
        let all_zero = |v: &[f32]| v.iter().all(|&x| x == 0.0);
        for (sid, e) in substrate.all_streams() {
            if !matches!(e.decl, Some(StreamDecl::Turn(_))) {
                continue;
            }
            let Some(hist) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
                continue;
            };
            let full = hist.first().map(|s| s.n_heads as u32 * 128).unwrap_or(0);
            let dead: Vec<usize> = hist
                .iter()
                .enumerate()
                .filter(|(_, s)| s.popcount() == full || s.popcount() == 0)
                .map(|(i, _)| i)
                .collect();
            if dead.is_empty() {
                continue;
            }
            let Some(ids) = persistence
                .read_tokens(&substrate, sid)
                .ok()
                .flatten()
                .and_then(|b| decode_token_ids(&b).ok())
            else {
                continue;
            };
            let text = tok.decode(&ids, false).unwrap_or_default();
            if !want.is_empty() && !text.contains(&want) {
                continue;
            }
            let n_tok = ids.len();
            let chunks = persistence
                .read_stream_chunks(&substrate, sid)
                .unwrap_or_default();
            let n_chunks = chunks.len();
            if n_chunks == 0 || n_chunks % N_LAYERS != 0 {
                println!(
                    "stream {}: {n_chunks} chunks not divisible by {N_LAYERS}",
                    sid.0
                );
                continue;
            }
            let cpl = n_chunks / N_LAYERS;
            let mut by_idx: HashMap<u64, &ChunkPayload> = HashMap::new();
            for (ci, p) in &chunks {
                by_idx.insert(*ci, p);
            }
            println!(
                "\n=== CHUNKLAYOUT stream {}  n_tok={n_tok}  wideQ={}  dead={}  chunks/layer={cpl} ===",
                sid.0,
                hist.len(),
                dead.len()
            );
            println!(
                "  (layer-0 chunks; X=nonzero-K slot, .=zero-K slot; cum = block×32 position)"
            );
            let mut real_tokens = 0usize;
            for b in 0..cpl {
                let Some(payload) = by_idx.get(&(b as u64)) else {
                    println!("  block {b:>2}: MISSING");
                    continue;
                };
                let Some((_q, k)) = parse_r16_chunk(&payload.kv_bytes) else {
                    println!(
                        "  block {b:>2}: off={:>2}  NON-R16 ({} B)",
                        payload.offset,
                        payload.kv_bytes.len()
                    );
                    continue;
                };
                let mut map = String::new();
                let mut fill = 0usize;
                for t in 0..32 {
                    let z = all_zero(&token_k(&k, t));
                    map.push(if z { '.' } else { 'X' });
                    if !z {
                        fill += 1;
                    }
                }
                real_tokens += fill;
                println!(
                    "  block {b:>2}: off={:>2} fill={fill:>2}/32  cum {:>4}..{:<4} [{map}]",
                    payload.offset,
                    b * 32,
                    b * 32 + 32
                );
            }
            println!(
                "  SUM real (nonzero-K) slots = {real_tokens}   vs Tokens-record = {n_tok}   vs wide-Q = {}",
                hist.len()
            );
            println!(
                "  dead wide-Q (block:offset): {}",
                dead.iter()
                    .take(50)
                    .map(|&i| format!("{}:{}", i / 32, i % 32))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            return Ok(());
        }
        println!("(no dead-Q turn matched)");
        return Ok(());
    }

    let m = Markers {
        im_start: tok.token_to_id("<|im_start|>").expect("im_start"),
        im_end: tok.token_to_id("<|im_end|>").expect("im_end"),
        think_end: tok.token_to_id("</think>"),
    };

    // ── Tool-def sections: first 12 tokens, mean-pooled, L24–40 (one+ copy/tool) ─
    let def_streams: Vec<(StreamId, String)> = substrate
        .all_streams()
        .filter_map(|(sid, e)| match &e.decl {
            Some(StreamDecl::PromptSection(d)) => Some((sid, d.debug_name.clone())),
            _ => None,
        })
        .filter(|(_, n)| {
            // Tool-definition sections only: drop the system-prompt framing sections
            // (frame, reasoning_stance, grounding, …) that share the alphabetic-name
            // shape — they are NOT tool defs and the attention sink makes `frame`
            // dominate every dot product. THIS FILTER IS WHY §21–§29 NOW WORK:
            // before it, `frame` (attention sink, positions 0–3) scored ~8700 vs ~1600
            // for any real tool and won on essentially every probe, pinning every
            // call→def readout at chance. The scan was ranking the system scaffold, not
            // the definitions. See §30 (the diagnostic that found it) and docs §21.2.
            let base = n.split(':').next().unwrap_or(n);
            n.as_bytes()
                .first()
                .map(|b| b.is_ascii_alphabetic())
                .unwrap_or(false)
                && !n.starts_with("section_")
                && !matches!(
                    base,
                    "frame"
                        | "reasoning_stance"
                        | "grounding"
                        | "history_stance"
                        | "tools_overview"
                        | "thinking_effort"
                        | "response_length"
                )
        })
        .collect();
    let mut defs: Vec<(String, Vec<f32>)> = Vec::new();
    let mut def_mw: Vec<(String, Vec<f32>)> = Vec::new(); // mean-whole-filtered def
    let mut def_tok: Vec<(String, Vec<Vec<f32>>)> = Vec::new(); // §29: per-token def K (first SWIN)
    let mut def_sign: Vec<(String, Vec<Vec<u64>>)> = Vec::new(); // §31: ALL def tokens, sign-packed
    let mut def_content: Vec<(String, Vec<f32>)> = Vec::new(); // §49: ALWAYS content-only mean K
    let mut tool_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (sid, name) in &def_streams {
        let Some(tbytes) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
            continue;
        };
        let Ok(toks) = decode_token_ids(&tbytes) else {
            continue;
        };
        let Some(tb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
            continue;
        };
        defs.push((name.clone(), mean_band(&tb, 0..SWIN.min(toks.len()))));
        let whole: Vec<usize> = (0..toks.len())
            .filter(|&t| t < tb.len() && !is_structural(&tok, toks[t]))
            .collect();
        let w = if whole.len() >= 3 {
            whole
        } else {
            (0..tb.len()).collect()
        };
        def_mw.push((name.clone(), mean_of(&tb, &w)));
        // Content-only mean K (structural rules ALWAYS applied, ignoring KEEP_STRUCT) —
        // a sharp def-def similarity basis for §49 even when scoring keeps all tokens.
        let content: Vec<usize> = (0..toks.len())
            .filter(|&t| t < tb.len() && !is_structural_rules(&tok, toks[t]))
            .collect();
        let cw = if content.len() >= 3 {
            content
        } else {
            (0..tb.len()).collect()
        };
        def_content.push((name.clone(), mean_of(&tb, &cw)));
        def_tok.push((name.clone(), tb.iter().take(SWIN).cloned().collect()));
        // §31–§45 score the call against ONLY the def's non-structural tokens (`w`),
        // dropping the shared JSON/markup scaffold from the definition side too.
        def_sign.push((name.clone(), w.iter().map(|&t| sign_pack(&tb[t])).collect()));
        tool_set.insert(name.clone());
    }
    eprintln!(
        "tool defs: {} copies over {} tools",
        defs.len(),
        tool_set.len()
    );

    // ── Tool-call probes: first 8 tokens of the call, PER-TOKEN, L24–40 ─────────
    let mut turn_streams: Vec<StreamId> = substrate
        .all_streams()
        .filter_map(|(sid, e)| match &e.decl {
            Some(StreamDecl::Turn(_)) => Some(sid),
            _ => None,
        })
        .collect();
    turn_streams.sort_by_key(|s| s.0);
    {
        // group by timeline; the highest turn_index stream holds the full conversation
        let mut tl: HashMap<u64, (u32, StreamId)> = HashMap::new();
        for (sid, e) in substrate.all_streams() {
            if let Some(StreamDecl::Turn(t)) = &e.decl {
                let ent = tl.entry(t.timeline_id).or_insert((t.turn_index, sid));
                if t.turn_index >= ent.0 {
                    *ent = (t.turn_index, sid);
                }
            }
        }
        let (mut tool_c, mut indep_c, mut shown) = (0usize, 0usize, 0usize);
        for (_tlid, (_ti, sid)) in &tl {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let msgs = parse_messages(&toks, &m, &tok);
            let text: String = toks.iter().map(|&t| detok(&tok, t)).collect();
            let is_tool = text.contains("<tool_call>") || text.contains("tool_call");
            if is_tool {
                tool_c += 1;
            } else {
                indep_c += 1;
            }
            if !is_tool && shown < 5 {
                let roles: String = msgs
                    .iter()
                    .map(|x| match x.role {
                        Role::System => 'S',
                        Role::User => 'U',
                        Role::Assistant => 'A',
                        _ => '?',
                    })
                    .collect();
                let head: String = toks.iter().take(24).map(|&t| detok(&tok, t)).collect();
                eprintln!(
                    "  indep[{}]: {} msgs [{}]  «{}»",
                    shown,
                    msgs.len(),
                    roles,
                    head.replace('\n', "⏎").chars().take(80).collect::<String>()
                );
                shown += 1;
            }
        }
        eprintln!(
            "TIMELINES: {} total → {} tool-call, {} independent conversations",
            tl.len(),
            tool_c,
            indep_c
        );
    }
    // (tool, per-token bands of the 8-token call window)
    let mut probes: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    let mut probe_mw: Vec<(String, Vec<f32>)> = Vec::new(); // mean-whole-filtered call
    let mut probe_win: Vec<(String, Vec<f32>)> = Vec::new(); // §16 name-window mean (memory lane)
    let mut tool_phase: [Vec<Vec<f32>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()]; // §21.2 phases: user / user+think / think+asst / asst
    let mut tool_phase_tool: Vec<String> = Vec::new();
    // §21.2 per-token sign packs (K and Q) + the four phase token ranges, per case —
    // for the per-token MAX readout and Q·K fusion (the mean-pooled `tool_phase`
    // above is kept for §20 / §21.1).
    let mut tool_k_sigs: Vec<Vec<Vec<u64>>> = Vec::new();
    let mut tool_q_sigs: Vec<Vec<Vec<u64>>> = Vec::new();
    let mut tool_ranges: Vec<[Vec<usize>; 4]> = Vec::new();
    // §27: per-phase summed Q/K float band per case — `Σ_{token∈phase} band`, so the
    // full-precision attention score Σ_t (Q_t·K_def) = (Σ_t Q_t)·K_def is one dot.
    let mut tool_q_phase: Vec<[Vec<f32>; 4]> = Vec::new();
    let mut tool_k_phase: Vec<[Vec<f32>; 4]> = Vec::new();
    // §28: every token's Q/K float band (not pooled) — for real per-token attention.
    let mut tool_q_float: Vec<Vec<Vec<f32>>> = Vec::new();
    let (mut n_asst, mut n_named, mut n_matched, mut n_banded, mut dbg) = (0usize, 0, 0, 0, 0);
    for sid in &turn_streams {
        let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
            continue;
        };
        let Ok(toks) = decode_token_ids(&tb) else {
            continue;
        };
        let msgs = parse_messages(&toks, &m, &tok);
        if !msgs.iter().any(|x| x.role == Role::Assistant) {
            continue;
        }
        n_asst += 1;
        // scan EVERY assistant message for a tool_call naming a known tool. The call
        // sits at the assistant START (driver prefills `<tool_call>`, and even a
        // natural call is the first thing emitted) — §16's asst_start+8 window.
        let mut found: Option<(String, usize, usize)> = None;
        let mut any_text = String::new();
        for a in msgs.iter().filter(|x| x.role == Role::Assistant) {
            let text: String = (a.start..a.end)
                .filter(|&t| t < toks.len())
                .map(|t| detok(&tok, toks[t]))
                .collect();
            if any_text.is_empty() {
                any_text = text.clone();
            }
            if let Some(tool) = tool_name_from_text(&text) {
                if tool_set.contains(&tool) {
                    found = Some((tool, a.start, a.end));
                    break;
                }
            }
        }
        let Some((tool, call_start, a_end)) = found else {
            if dbg < 8 {
                eprintln!(
                    "    [no-call] {:?}",
                    any_text.chars().take(90).collect::<String>()
                );
                dbg += 1;
            }
            continue;
        };
        n_named += 1;
        n_matched += 1;
        let Some(bands) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
            continue;
        };
        n_banded += 1;
        let win: Vec<Vec<f32>> = (call_start..(call_start + PWIN).min(a_end))
            .filter(|&t| t < bands.len())
            .map(|t| bands[t].clone())
            .collect();
        if win.is_empty() {
            continue;
        }
        let cwhole: Vec<usize> = (call_start..a_end)
            .filter(|&t| t < bands.len() && !is_structural(&tok, toks[t]))
            .collect();
        let cw = if cwhole.len() >= 2 {
            cwhole
        } else {
            (call_start..a_end.min(bands.len())).collect()
        };
        probe_mw.push((tool.clone(), mean_of(&bands, &cw)));
        let widx: Vec<usize> = (call_start..(call_start + PWIN).min(a_end))
            .filter(|&t| t < bands.len())
            .collect();
        probe_win.push((tool.clone(), mean_of(&bands, &widx)));
        probes.push((tool, win));
    }
    eprintln!(
        "probe funnel: {} turns → {} w/asst → {} named → {} matched-def → {} banded → {} probes",
        turn_streams.len(),
        n_asst,
        n_named,
        n_matched,
        n_banded,
        probes.len()
    );
    {
        let mut per: HashMap<&str, usize> = HashMap::new();
        for (t, _) in &probes {
            *per.entry(t.as_str()).or_default() += 1;
        }
        let multi = per.values().filter(|&&c| c >= 2).count();
        eprintln!(
            "  tools with ≥2 probes: {} / {} (mean {:.2} probes/tool)",
            multi,
            per.len(),
            probes.len() as f64 / per.len().max(1) as f64
        );
    }

    // ── tool-conversation 3-phase parse — reconstruct each FULL conversation from its
    //    timeline (user turn + assistant turn are separate streams), then apply the
    //    IDENTICAL phase recipe as the corpus: user / user+think / think+call.
    {
        let mut timelines: HashMap<u64, Vec<(u32, StreamId)>> = HashMap::new();
        for (sid, e) in substrate.all_streams() {
            if let Some(StreamDecl::Turn(t)) = &e.decl {
                timelines
                    .entry(t.timeline_id)
                    .or_default()
                    .push((t.turn_index, sid));
            }
        }
        let (mut c_toks, mut c_found, mut c_ci, mut c_u) = (0usize, 0usize, 0usize, 0usize);
        // Deterministic case order: iterate timelines sorted by id (the HashMap's
        // own order varies run-to-run, which made `case 0` a different tool each run).
        let mut tl_sorted: Vec<(&u64, &Vec<(u32, StreamId)>)> = timelines.iter().collect();
        tl_sorted.sort_by_key(|(tlid, _)| **tlid);
        for (_tlid, turns) in tl_sorted {
            let mut turns = turns.clone();
            turns.sort_by_key(|x| x.0);
            let mut all_toks: Vec<u32> = Vec::new();
            let mut all_kbands: Vec<Vec<f32>> = Vec::new();
            let mut all_qbands: Vec<Vec<f32>> = Vec::new();
            for (_ti, sid) in &turns {
                let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                    continue;
                };
                let Ok(toks) = decode_token_ids(&tb) else {
                    continue;
                };
                // Probe chunks are R16 (quantization disabled) → both Q and K present.
                let Some((qbands, kbands)) =
                    read_token_qk_bands(&mut persistence, &substrate, *sid, toks.len())
                else {
                    continue;
                };
                let n = toks.len().min(kbands.len());
                all_toks.extend_from_slice(&toks[..n]);
                all_kbands.extend(kbands.into_iter().take(n));
                all_qbands.extend(qbands.into_iter().take(n));
            }
            if all_toks.is_empty() {
                continue;
            }
            c_toks += 1;
            // The prompt has no <|im_start|> marker (parse_messages drops it); the FIRST
            // <|im_start|> begins the assistant turn. Everything before = the user prompt.
            let Some(asst_start) = all_toks.iter().position(|&t| t == m.im_start) else {
                continue;
            };
            let nb = all_toks.len().min(all_kbands.len());
            if asst_start == 0 || asst_start >= nb {
                continue;
            }
            let asst_text: String = (asst_start..all_toks.len())
                .map(|t| detok(&tok, all_toks[t]))
                .collect();
            let Some(tool) = tool_name_from_text(&asst_text).filter(|t| tool_set.contains(t))
            else {
                continue;
            };
            c_found += 1;
            c_ci += 1;
            let resp_start = match m.think_end {
                Some(t1) => (asst_start..nb)
                    .find(|&i| all_toks[i] == t1)
                    .map(|e| e + 1)
                    .unwrap_or(asst_start),
                None => asst_start,
            };
            let u: Vec<usize> = (0..asst_start.min(nb))
                .filter(|&t| !is_structural(&tok, all_toks[t]))
                .collect();
            let th: Vec<usize> = (asst_start..resp_start.min(nb))
                .filter(|&t| !is_structural(&tok, all_toks[t]))
                .collect();
            let r: Vec<usize> = (resp_start.min(nb)..nb)
                .filter(|&t| !is_structural(&tok, all_toks[t]))
                .collect();
            if u.is_empty() {
                continue;
            }
            c_u += 1;
            let mut ut = u.clone();
            ut.extend(&th);
            let mut tr = th.clone();
            tr.extend(&r);
            tool_phase[0].push(mean_of(&all_kbands, &u));
            tool_phase[1].push(mean_of(&all_kbands, &ut));
            tool_phase[2].push(mean_of(&all_kbands, &tr));
            tool_phase[3].push(mean_of(&all_kbands, &r));
            tool_phase_tool.push(tool);
            // §21.2: per-token sign packs (K and Q) over all tokens + the 4 ranges.
            tool_k_sigs.push(all_kbands.iter().map(|b| sign_pack(b)).collect());
            tool_q_sigs.push(all_qbands.iter().map(|b| sign_pack(b)).collect());
            // §27: sum each phase's per-token float bands once (full-precision attention).
            let phase_idx: [&Vec<usize>; 4] = [&u, &ut, &tr, &r];
            let bdim = all_kbands.first().map(|b| b.len()).unwrap_or(0);
            tool_q_phase.push(std::array::from_fn(|pi| {
                let mut s = vec![0f32; bdim];
                for &t in phase_idx[pi].iter() {
                    for (d, &x) in all_qbands[t].iter().enumerate() {
                        s[d] += x;
                    }
                }
                s
            }));
            tool_k_phase.push(std::array::from_fn(|pi| {
                let mut s = vec![0f32; bdim];
                for &t in phase_idx[pi].iter() {
                    for (d, &x) in all_kbands[t].iter().enumerate() {
                        s[d] += x;
                    }
                }
                s
            }));
            tool_q_float.push(all_qbands.clone());
            tool_ranges.push([u.clone(), ut.clone(), tr.clone(), r.clone()]);
        }
        eprintln!("tool 3-phase: {} conversations reconstructed (from {} timelines); toks={} found={} ci>0={} u_ok={}", tool_phase_tool.len(), timelines.len(), c_toks, c_found, c_ci, c_u);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §21 — RESTORED HOLDOUT TEST (three-phase) over the tool cases.
    //
    //  This block is corpus-independent — it trains and tests purely on the tool
    //  conversations already in the substrate. Set `S21_ONLY=1` to stop here and
    //  skip the corpus load + §16–§20 transfer experiments.
    //
    //  Step 1 — the POSITIVE SET. Match every reconstructed test case to its tool
    //  definition section in the substrate. Each case's in-flight `tool_call` names
    //  a tool; that tool's definition is a pinned `def_streams` section — the
    //  routing target. A case is a routable POSITIVE iff its tool's definition is
    //  present. This is the ground truth the three-phase routing scores against.
    // ════════════════════════════════════════════════════════════════════════
    println!("\n══ §21.1 — holdout test-case → tool-definition matching (the positive set) ══");
    {
        // tool name → its definition-section stream id(s) in the substrate.
        let mut def_ids: HashMap<&str, Vec<StreamId>> = HashMap::new();
        for (sid, name) in &def_streams {
            def_ids.entry(name.as_str()).or_default().push(*sid);
        }
        // tool name → number of reconstructed test cases that call it.
        let mut cases_per_tool: HashMap<&str, usize> = HashMap::new();
        for t in &tool_phase_tool {
            *cases_per_tool.entry(t.as_str()).or_default() += 1;
        }
        let mut rows: Vec<(&str, usize, Vec<StreamId>)> = cases_per_tool
            .iter()
            .map(|(&t, &n)| (t, n, def_ids.get(t).cloned().unwrap_or_default()))
            .collect();
        // most-exercised tools first, then alphabetical for a stable read.
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

        println!(
            "  {} reconstructed cases · {} distinct tools called · matched against {} def sections",
            tool_phase_tool.len(),
            rows.len(),
            def_streams.len(),
        );
        println!(
            "  {:<30} {:>5} {:>4}  {:<8}  {}",
            "tool", "cases", "defs", "status", "def section stream id(s)"
        );
        let (mut pos_cases, mut neg_cases, mut pos_tools) = (0usize, 0usize, 0usize);
        for (t, n, ids) in &rows {
            let ok = !ids.is_empty();
            if ok {
                pos_cases += n;
                pos_tools += 1;
            } else {
                neg_cases += n;
            }
            let idstr = if ids.is_empty() {
                "— (NO DEFINITION IN SUBSTRATE)".to_string()
            } else {
                let head = ids
                    .iter()
                    .take(3)
                    .map(|s| format!("0x{:016x}", s.0))
                    .collect::<Vec<_>>()
                    .join(" ");
                if ids.len() > 3 {
                    format!("{head} (+{})", ids.len() - 3)
                } else {
                    head
                }
            };
            println!(
                "  {:<30} {:>5} {:>4}  {:<8}  {}",
                t,
                n,
                ids.len(),
                if ok { "POSITIVE" } else { "MISSING!" },
                idstr
            );
        }
        println!("  ───────────────────────────────────────────────────────────────");
        println!(
            "  POSITIVE SET: {}/{} cases routable · {} tools with a def · {} cases with NO def",
            pos_cases,
            tool_phase_tool.len(),
            pos_tools,
            neg_cases
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §21.2 — four-phase raw-sign routing, with (A) per-token MAX readout and
    //  (C) Q·K + K·K fusion.  [STABLE baseline — §22 iterates the SUM variant; if that
    //  doesn't beat this, we stay here.]
    //
    //  For each case × 4 query phases, every probe token contributes sign(K_t) and
    //  sign(Q_t) (raw — no rotation). Each token is XOR-popcount scored vs every tool
    //  definition's sign(K), and a def's score is the **MAX of those per-token pop
    //  values over the phase's tokens** (soft-attention readout). The K·K and Q·K
    //  channels are each z-normalised across defs, blended **0.4·Q·K + 0.6·K·K**
    //  (§7's fusion recipe), folded by tool (max over copies), ranked. A probe is
    //  POSITIVE if its correct def lands Top-1 (and Top-5), reported per phase.
    //  Layers are fixed per phase (full routing band).
    //
    //  RESULT: ~chance (best Top-5 7.5%, chance 5.2%). This is the start of the
    //  raw-sign "back to basics" arc — without the §16–§20 CCA rotation, call→def is
    //  the §4 domain gap. §22/§25/§26/§27–§29 confirm chance under every readout; §30
    //  finds the real culprit (system sections in the def list); §31 rebuilds it on
    //  clean defs with a weighted atom. See docs §21.1.
    // ════════════════════════════════════════════════════════════════════════
    println!("\n══ §21.2 — four-phase raw-sign routing: per-token MAX readout + (0.4·Q·K + 0.6·K·K) fusion ══");
    {
        let phase_sel: [Option<Sel>; 4] = [None, None, None, None];
        let phase_names = ["user", "user+think", "think+asst", "assistant"];
        let layer_list = |sel: Option<Sel>| -> String {
            let layers: Vec<usize> = match sel {
                None | Some(Sel::Head(_)) => (BAND_LO..BAND_HI).collect(),
                Some(Sel::Layer(l)) => vec![BAND_LO + l],
            };
            layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        // Sign-pack each def's mean K once (the routing targets); fold by tool below.
        let def_packs: Vec<(&str, Vec<u64>)> = def_mw
            .iter()
            .map(|(t, b)| (t.as_str(), sign_pack(b)))
            .collect();
        let dims = def_mw.first().map(|(_, b)| b.len()).unwrap_or(0);
        let n_probes = tool_k_sigs.len();
        println!(
            "  {} cases × 4 phases = {} probes · rank vs {} tool defs · chance {:.1}% / {:.1}% (Top-1/Top-5)",
            n_probes,
            n_probes * 4,
            tool_set.len(),
            100.0 / tool_set.len() as f64,
            500.0 / tool_set.len() as f64,
        );
        println!(
            "  {:<14} {:>8} {:>8}   {}",
            "phase", "Top-1%", "Top-5%", "layers"
        );
        for p in 0..4 {
            let mut acc = Acc::default();
            for i in 0..n_probes {
                let range = &tool_ranges[i][p];
                if range.is_empty() {
                    acc.add(None);
                    continue;
                }
                // Score each def = MAX over the phase's tokens of the per-token
                // XOR-popcount match (each token's whole-band sign(K) vs the def's
                // sign(K)) — the soft-attention readout.
                let mut kk = vec![f32::MIN; def_packs.len()];
                let mut qk = vec![f32::MIN; def_packs.len()];
                for (d, (_t, dp)) in def_packs.iter().enumerate() {
                    for &tk in range {
                        kk[d] = kk[d].max(sign_match(&tool_k_sigs[i][tk], dp, dims));
                        qk[d] = qk[d].max(sign_match(&tool_q_sigs[i][tk], dp, dims));
                    }
                }
                // z-normalise each channel across defs, blend, fold copies by tool.
                let zkk = znorm(&kk);
                let zqk = znorm(&qk);
                let mut per: HashMap<&str, f32> = HashMap::new();
                for (d, (t, _)) in def_packs.iter().enumerate() {
                    let s = 0.4 * zqk[d] + 0.6 * zkk[d];
                    let e = per.entry(*t).or_insert(f32::MIN);
                    *e = e.max(s);
                }
                acc.add(rank_tool(&per, tool_phase_tool[i].as_str()));
            }
            println!(
                "  {:<14} {:>7.1}% {:>7.1}%   {}",
                phase_names[p],
                100.0 * acc.t1 as f64 / acc.n.max(1) as f64,
                100.0 * acc.t5 as f64 / acc.n.max(1) as f64,
                layer_list(phase_sel[p]),
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §22 — ITERATION on §21.2: per-token SUM readout (instead of MAX).
    //
    //  Identical to §21.2 except a def's score SUMS the per-token XOR-popcount
    //  matches over the phase's tokens rather than taking their max — the "sum of the
    //  pop values" combine. Kept as its own section to iterate on (next: rotation on
    //  top); if it does not beat §21.2's MAX, we drop back to §21.
    // ════════════════════════════════════════════════════════════════════════
    println!("\n══ §22 — four-phase raw-sign routing: per-token SUM readout + (0.4·Q·K + 0.6·K·K) fusion ══");
    {
        let phase_sel: [Option<Sel>; 4] = [None, None, None, None];
        let phase_names = ["user", "user+think", "think+asst", "assistant"];
        let layer_list = |sel: Option<Sel>| -> String {
            let layers: Vec<usize> = match sel {
                None | Some(Sel::Head(_)) => (BAND_LO..BAND_HI).collect(),
                Some(Sel::Layer(l)) => vec![BAND_LO + l],
            };
            layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let def_packs: Vec<(&str, Vec<u64>)> = def_mw
            .iter()
            .map(|(t, b)| (t.as_str(), sign_pack(b)))
            .collect();
        let dims = def_mw.first().map(|(_, b)| b.len()).unwrap_or(0);
        let n_probes = tool_k_sigs.len();
        println!(
            "  {} cases × 4 phases = {} probes · rank vs {} tool defs · chance {:.1}% / {:.1}% (Top-1/Top-5)",
            n_probes,
            n_probes * 4,
            tool_set.len(),
            100.0 / tool_set.len() as f64,
            500.0 / tool_set.len() as f64,
        );
        println!(
            "  {:<14} {:>8} {:>8}   {}",
            "phase", "Top-1%", "Top-5%", "layers"
        );
        for p in 0..4 {
            let mut acc = Acc::default();
            for i in 0..n_probes {
                let range = &tool_ranges[i][p];
                if range.is_empty() {
                    acc.add(None);
                    continue;
                }
                // SUM over the phase's tokens of the per-token XOR-popcount match.
                let mut kk = vec![0.0f32; def_packs.len()];
                let mut qk = vec![0.0f32; def_packs.len()];
                for (d, (_t, dp)) in def_packs.iter().enumerate() {
                    for &tk in range {
                        kk[d] += sign_match(&tool_k_sigs[i][tk], dp, dims);
                        qk[d] += sign_match(&tool_q_sigs[i][tk], dp, dims);
                    }
                }
                let zkk = znorm(&kk);
                let zqk = znorm(&qk);
                let mut per: HashMap<&str, f32> = HashMap::new();
                for (d, (t, _)) in def_packs.iter().enumerate() {
                    let s = 0.4 * zqk[d] + 0.6 * zkk[d];
                    let e = per.entry(*t).or_insert(f32::MIN);
                    *e = e.max(s);
                }
                acc.add(rank_tool(&per, tool_phase_tool[i].as_str()));
            }
            println!(
                "  {:<14} {:>7.1}% {:>7.1}%   {}",
                phase_names[p],
                100.0 * acc.t1 as f64 / acc.n.max(1) as f64,
                100.0 * acc.t5 as f64 / acc.n.max(1) as f64,
                layer_list(phase_sel[p]),
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §23 — DISTRIBUTION of per-head match (SUM approach): correct vs incorrect.
    //
    //  Break the §22 K·K SUM score down PER HEAD. For each (probe-phase, def, head)
    //  the per-head match fraction = mean over the phase's tokens of
    //  (head_dims − XOR-popcount)/head_dims ∈ [0,1] (0.5 = chance, higher ⇒ closer).
    //  Split by whether the def's tool matches the probe's tool, and compare the
    //  correct vs incorrect distributions per head — does a correct def's head value
    //  sit apart from an incorrect one? Cohen's d quantifies the separation per
    //  (phase, head). (K·K channel; Q·K is symmetric and available the same way.)
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "\n══ §23 — per-head K·K match distribution: correct vs incorrect defs (SUM approach) ══"
    );
    {
        let def_packs: Vec<(&str, Vec<u64>)> = def_mw
            .iter()
            .map(|(t, b)| (t.as_str(), sign_pack(b)))
            .collect();
        let head_dims = (BAND_HI - BAND_LO) * HEAD_DIM; // 2048
        let phase_names = ["user", "user+think", "think+asst", "assistant"];
        let mut correct: [Vec<f32>; N_KV_HEAD] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut incorrect: [Vec<f32>; N_KV_HEAD] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut d_table = [[0f32; N_KV_HEAD]; 4];
        for p in 0..4 {
            let mut pc: [Vec<f32>; N_KV_HEAD] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut pi: [Vec<f32>; N_KV_HEAD] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for i in 0..tool_k_sigs.len() {
                let range = &tool_ranges[i][p];
                if range.is_empty() {
                    continue;
                }
                let t = range.len() as f32;
                let truth = tool_phase_tool[i].as_str();
                for (dt, dp) in &def_packs {
                    for h in 0..N_KV_HEAD {
                        let mut s = 0f32;
                        for &tk in range {
                            s += (head_dims as u32 - xor_pop_head(&tool_k_sigs[i][tk], dp, h))
                                as f32;
                        }
                        let frac = s / (t * head_dims as f32);
                        if *dt == truth {
                            pc[h].push(frac);
                            correct[h].push(frac);
                        } else {
                            pi[h].push(frac);
                            incorrect[h].push(frac);
                        }
                    }
                }
            }
            for h in 0..N_KV_HEAD {
                d_table[p][h] = cohens_d(&pc[h], &pi[h]);
            }
        }
        println!(
            "  metric: per-head match fraction = mean_token (matching bits / {head_dims}); 0.5 = chance, higher ⇒ closer"
        );
        println!("\n  cohen's d (correct − incorrect, in pooled std) by phase × head:");
        println!(
            "  {:<14} {:>8} {:>8} {:>8} {:>8}",
            "phase", "head0", "head1", "head2", "head3"
        );
        for p in 0..4 {
            println!(
                "  {:<14} {:>+8.3} {:>+8.3} {:>+8.3} {:>+8.3}",
                phase_names[p], d_table[p][0], d_table[p][1], d_table[p][2], d_table[p][3]
            );
        }
        println!("\n  pooled over all phases — per-head distribution (percentiles of the match fraction):");
        for h in 0..N_KV_HEAD {
            let mut c = correct[h].clone();
            let mut inc = incorrect[h].clone();
            c.sort_by(|a, b| a.partial_cmp(b).unwrap());
            inc.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (cm, cs) = mean_std(&c);
            let (im, is) = mean_std(&inc);
            println!("  head {h}   d={:+.3}", cohens_d(&c, &inc));
            println!(
                "    correct   n={:<6} mean={:.4}±{:.4}  p10={:.4} p50={:.4} p90={:.4}",
                c.len(),
                cm,
                cs,
                pct_of(&c, 0.10),
                pct_of(&c, 0.50),
                pct_of(&c, 0.90)
            );
            println!(
                "    incorrect n={:<6} mean={:.4}±{:.4}  p10={:.4} p50={:.4} p90={:.4}",
                inc.len(),
                im,
                is,
                pct_of(&inc, 0.10),
                pct_of(&inc, 0.50),
                pct_of(&inc, 0.90)
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §26 — three-level aggregation formula scan (head → layer → across-layers).
    //
    //  §22's score is sum/sum/sum: sum the 128 per-dim sign matches into a head,
    //  sum a layer's 4 heads, sum the 16 layers. §26 scans alternative reductions at
    //  each of the three levels independently and reports the best combination per
    //  phase + the overall top combos. Per-dim match counts (summed over the phase's
    //  tokens) feed the HEAD formula; head values feed the LAYER formula; layer
    //  values feed the ACROSS formula; the result is z-normed per def, fused
    //  (0.4·Q·K + 0.6·K·K), folded per tool (max over copies), ranked. Run `S26=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S26").is_ok() {
        #[derive(Clone, Copy)]
        enum Red {
            Sum,
            Max,
            Min,
            PowMean(f32),
            TopK(usize),
        }
        impl Red {
            fn apply(&self, xs: &[f32]) -> f32 {
                match self {
                    Red::Sum => xs.iter().sum(),
                    Red::Max => xs.iter().cloned().fold(f32::MIN, f32::max),
                    Red::Min => xs.iter().cloned().fold(f32::MAX, f32::min),
                    Red::PowMean(p) => {
                        let n = xs.len().max(1) as f32;
                        (xs.iter().map(|&x| x.max(0.0).powf(*p)).sum::<f32>() / n).powf(1.0 / p)
                    }
                    Red::TopK(k) => {
                        let mut v = xs.to_vec();
                        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        v.iter().take(*k).sum()
                    }
                }
            }
            fn label(&self) -> String {
                match self {
                    Red::Sum => "sum".to_string(),
                    Red::Max => "max".to_string(),
                    Red::Min => "min".to_string(),
                    Red::PowMean(p) => format!("pmean{:.0}", p),
                    Red::TopK(k) => format!("top{k}"),
                }
            }
        }

        let head_formulas = [
            Red::Sum,
            Red::Max,
            Red::Min,
            Red::PowMean(2.0),
            Red::PowMean(4.0),
            Red::TopK(16),
            Red::TopK(32),
            Red::TopK(64),
        ];
        let layer_formulas = [
            Red::Sum,
            Red::Max,
            Red::Min,
            Red::PowMean(2.0),
            Red::TopK(2),
        ];
        let across_formulas = [
            Red::Sum,
            Red::Max,
            Red::Min,
            Red::PowMean(2.0),
            Red::TopK(4),
            Red::TopK(8),
        ];

        let def_packs: Vec<(&str, Vec<u64>)> = def_mw
            .iter()
            .map(|(t, b)| (t.as_str(), sign_pack(b)))
            .collect();
        let dims = def_mw.first().map(|(_, b)| b.len()).unwrap_or(0);
        let n_defs = def_packs.len();
        let n_probes = tool_k_sigs.len();
        let n_layers = BAND_HI - BAND_LO; // 16
        let n_heads = n_layers * N_KV_HEAD; // 64
        let words = dims.div_ceil(64);
        let phase_names = ["user", "user+think", "think+asst", "assistant"];
        let n_combos = head_formulas.len() * layer_formulas.len() * across_formulas.len();

        println!("\n══ §26 — head→layer→across aggregation formula scan ══");
        println!(
            "  {} cases × 4 phases · rank vs {} tool defs · chance {:.1}% / {:.1}% (Top-1/Top-5) · {} combos/phase",
            n_probes,
            tool_set.len(),
            100.0 / tool_set.len() as f64,
            500.0 / tool_set.len() as f64,
            n_combos,
        );
        println!(
            "  head ∈ {:?}",
            head_formulas.iter().map(|r| r.label()).collect::<Vec<_>>()
        );
        println!(
            "  layer ∈ {:?}",
            layer_formulas.iter().map(|r| r.label()).collect::<Vec<_>>()
        );
        println!(
            "  across ∈ {:?}",
            across_formulas
                .iter()
                .map(|r| r.label())
                .collect::<Vec<_>>()
        );
        const K: usize = 5;
        println!(
            "  chance {:.1}/{:.1} (Top-1/Top-5)  ·  honest = {K}-fold CV: best combo on train probes → score held-out",
            100.0 / tool_set.len() as f64,
            500.0 / tool_set.len() as f64,
        );

        for p in 0..4 {
            // Precompute per-head reductions for every head formula, per (probe,def).
            let mut head_kk: Vec<Vec<f32>> =
                vec![vec![0f32; n_probes * n_defs * n_heads]; head_formulas.len()];
            let mut head_qk: Vec<Vec<f32>> =
                vec![vec![0f32; n_probes * n_defs * n_heads]; head_formulas.len()];
            for i in 0..n_probes {
                let range = &tool_ranges[i][p];
                if range.is_empty() {
                    continue;
                }
                for (di, (_t, dp)) in def_packs.iter().enumerate() {
                    // Per-dim match count, summed over the phase's tokens.
                    let mut mkk = vec![0f32; dims];
                    let mut mqk = vec![0f32; dims];
                    for &tk in range {
                        let pk = &tool_k_sigs[i][tk];
                        let pq = &tool_q_sigs[i][tk];
                        for w in 0..words {
                            let mk = !(pk[w] ^ dp[w]);
                            let mq = !(pq[w] ^ dp[w]);
                            let base = w * 64;
                            let bits = (dims - base).min(64);
                            for b in 0..bits {
                                let d = base + b;
                                mkk[d] += ((mk >> b) & 1) as f32;
                                mqk[d] += ((mq >> b) & 1) as f32;
                            }
                        }
                    }
                    let pd = (i * n_defs + di) * n_heads;
                    for hh in 0..n_heads {
                        let l = hh / N_KV_HEAD;
                        let h = hh % N_KV_HEAD;
                        let base = l * PER_LAYER_DIM + h * HEAD_DIM;
                        let hk = &mkk[base..base + HEAD_DIM];
                        let hq = &mqk[base..base + HEAD_DIM];
                        for (hi, hf) in head_formulas.iter().enumerate() {
                            head_kk[hi][pd + hh] = hf.apply(hk);
                            head_qk[hi][pd + hh] = hf.apply(hq);
                        }
                    }
                }
            }

            // Per-combo, per-(valid) probe Top-1/Top-5 hit, so the fold split can
            // select on one slice of probes and score on a disjoint slice. `valid` =
            // probes with tokens this phase.
            let valid: Vec<usize> = (0..n_probes)
                .filter(|&i| !tool_ranges[i][p].is_empty())
                .collect();
            let nv = valid.len().max(1) as f64;
            let mut hits1: Vec<Vec<bool>> = Vec::with_capacity(n_combos);
            let mut hits5: Vec<Vec<bool>> = Vec::with_capacity(n_combos);
            let mut labels: Vec<String> = Vec::with_capacity(n_combos);
            for (hi, hf) in head_formulas.iter().enumerate() {
                for lf in layer_formulas.iter() {
                    for af in across_formulas.iter() {
                        let mut h1 = vec![false; valid.len()];
                        let mut h5 = vec![false; valid.len()];
                        let mut lvk = vec![0f32; n_layers];
                        let mut lvq = vec![0f32; n_layers];
                        let mut hk4 = [0f32; N_KV_HEAD];
                        let mut hq4 = [0f32; N_KV_HEAD];
                        for (vi, &i) in valid.iter().enumerate() {
                            let mut kk = vec![0f32; n_defs];
                            let mut qk = vec![0f32; n_defs];
                            for di in 0..n_defs {
                                let pd = (i * n_defs + di) * n_heads;
                                for l in 0..n_layers {
                                    for h in 0..N_KV_HEAD {
                                        hk4[h] = head_kk[hi][pd + l * N_KV_HEAD + h];
                                        hq4[h] = head_qk[hi][pd + l * N_KV_HEAD + h];
                                    }
                                    lvk[l] = lf.apply(&hk4);
                                    lvq[l] = lf.apply(&hq4);
                                }
                                kk[di] = af.apply(&lvk);
                                qk[di] = af.apply(&lvq);
                            }
                            let zkk = znorm(&kk);
                            let zqk = znorm(&qk);
                            let mut per: HashMap<&str, f32> = HashMap::new();
                            for di in 0..n_defs {
                                let s = 0.4 * zqk[di] + 0.6 * zkk[di];
                                let e = per.entry(def_packs[di].0).or_insert(f32::MIN);
                                *e = e.max(s);
                            }
                            let r = rank_tool(&per, tool_phase_tool[i].as_str());
                            h1[vi] = r == Some(0);
                            h5[vi] = matches!(r, Some(x) if x < 5);
                        }
                        hits1.push(h1);
                        hits5.push(h5);
                        labels.push(format!(
                            "head={:<7} layer={:<7} across={}",
                            hf.label(),
                            lf.label(),
                            af.label()
                        ));
                    }
                }
            }
            let c5 = |c: usize| hits5[c].iter().filter(|&&b| b).count();
            let c1 = |c: usize| hits1[c].iter().filter(|&&b| b).count();
            // Baseline sum/sum/sum is a single fixed combo, so its full-set score is
            // already honest (nothing was selected).
            let base = labels
                .iter()
                .position(|l| {
                    l.starts_with("head=sum") && l.contains("layer=sum") && l.contains("across=sum")
                })
                .unwrap_or(0);
            let (base_t1, base_t5) = (100.0 * c1(base) as f64 / nv, 100.0 * c5(base) as f64 / nv);
            // In-sample max — the inflated best over all combos on the SAME probes.
            let is_t5 = 100.0 * (0..n_combos).map(c5).max().unwrap_or(0) as f64 / nv;
            // Honest: K-fold — pick the best combo on the train probes, score the
            // held-out probes. Selection sees only train; scoring only the held-out.
            let (mut cv1, mut cv5, mut cvn) = (0usize, 0usize, 0usize);
            let mut sel: HashMap<&str, usize> = HashMap::new();
            for f in 0..K {
                let mut best_c = 0usize;
                let mut best_tr = -1.0f64;
                for c in 0..n_combos {
                    let (mut h, mut n) = (0usize, 0usize);
                    for vi in 0..valid.len() {
                        if vi % K != f {
                            n += 1;
                            if hits5[c][vi] {
                                h += 1;
                            }
                        }
                    }
                    let tr = h as f64 / n.max(1) as f64;
                    if tr > best_tr {
                        best_tr = tr;
                        best_c = c;
                    }
                }
                for vi in 0..valid.len() {
                    if vi % K == f {
                        cvn += 1;
                        if hits5[best_c][vi] {
                            cv5 += 1;
                        }
                        if hits1[best_c][vi] {
                            cv1 += 1;
                        }
                    }
                }
                *sel.entry(labels[best_c].as_str()).or_insert(0) += 1;
            }
            let (cv_t1, cv_t5) = (
                100.0 * cv1 as f64 / cvn.max(1) as f64,
                100.0 * cv5 as f64 / cvn.max(1) as f64,
            );
            let mut selv: Vec<(&str, usize)> = sel.into_iter().collect();
            selv.sort_by(|a, b| b.1.cmp(&a.1));
            let top_sel = selv
                .first()
                .map(|(l, c)| format!("{l} ×{c}/{K}"))
                .unwrap_or_default();
            println!(
                "  {:<11} base {:>4.1}/{:>4.1}   in-sample-max T5 {:>4.1}   CV-selected {:>4.1}/{:>4.1}   [{}]",
                phase_names[p], base_t1, base_t5, is_t5, cv_t1, cv_t5, top_sel,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §27 — BASICS: full-precision attention routing (real Q·K / K·K dot products).
    //
    //  The irrefutable check. Instead of the sign-XOR proxy (§21.2/§22), score each
    //  def with the ACTUAL attention dot product: Σ over the phase's tokens of
    //  (probe Q · def K) = (Σ probe Q) · def K, and likewise K·K. Z-norm per def;
    //  rank by Q·K alone, K·K alone, and the 0.4/0.6 fusion. A single fixed formula
    //  (no selection) so the full-set Top-1/Top-5 is already honest. Run `S27=1`.
    //
    //  If real Q·K ranks the correct def well above chance → the signal is present
    //  and the sign proxy was discarding it. If it's at chance too → the raw call→def
    //  comparison carries no signal without the rotation (§16 CCA), or the captured
    //  probe Q / def K are not what we think.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S27").is_ok() {
        fn dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        }
        println!("\n══ §27 — full-precision attention routing (real Q·K and K·K dot products) ══");
        let n_probes = tool_q_phase.len();
        let phase_names = ["user", "user+think", "think+asst", "assistant"];
        println!(
            "  {} cases × 4 phases · rank vs {} tool defs · chance {:.1}% / {:.1}% (Top-1/Top-5)",
            n_probes,
            tool_set.len(),
            100.0 / tool_set.len() as f64,
            500.0 / tool_set.len() as f64,
        );
        println!(
            "  {:<14} {:>13} {:>13} {:>13}",
            "phase", "Q·K T1/T5", "K·K T1/T5", "fused T1/T5"
        );
        for p in 0..4 {
            let mut a_qk = Acc::default();
            let mut a_kk = Acc::default();
            let mut a_f = Acc::default();
            for i in 0..n_probes {
                if tool_ranges[i][p].is_empty() {
                    a_qk.add(None);
                    a_kk.add(None);
                    a_f.add(None);
                    continue;
                }
                let qsum = &tool_q_phase[i][p];
                let ksum = &tool_k_phase[i][p];
                let mut qk = vec![0f32; def_mw.len()];
                let mut kk = vec![0f32; def_mw.len()];
                for (d, (_t, db)) in def_mw.iter().enumerate() {
                    qk[d] = dot(qsum, db);
                    kk[d] = dot(ksum, db);
                }
                let zqk = znorm(&qk);
                let zkk = znorm(&kk);
                let rank = |scores: &[f32]| {
                    let mut per: HashMap<&str, f32> = HashMap::new();
                    for (d, (t, _)) in def_mw.iter().enumerate() {
                        let e = per.entry(t.as_str()).or_insert(f32::MIN);
                        *e = e.max(scores[d]);
                    }
                    rank_tool(&per, tool_phase_tool[i].as_str())
                };
                a_qk.add(rank(&zqk));
                a_kk.add(rank(&zkk));
                let fused: Vec<f32> = (0..def_mw.len())
                    .map(|d| 0.4 * zqk[d] + 0.6 * zkk[d])
                    .collect();
                a_f.add(rank(&fused));
            }
            let pct = |a: &Acc| {
                (
                    100.0 * a.t1 as f64 / a.n.max(1) as f64,
                    100.0 * a.t5 as f64 / a.n.max(1) as f64,
                )
            };
            let (q1, q5) = pct(&a_qk);
            let (k1, k5) = pct(&a_kk);
            let (f1, f5) = pct(&a_f);
            println!(
                "  {:<14} {:>5.1}/{:>5.1} {:>5.1}/{:>5.1} {:>5.1}/{:>5.1}",
                phase_names[p], q1, q5, k1, k5, f1, f5
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §28 — BASICS: real per-token attention, peak vs summed.
    //
    //  For each call token, the full-precision Q·K against each def (mean-pooled).
    //  `max-tok` = the PEAK match over the call's tokens (the single token that most
    //  attends to the def) — the sanity check that the call's Q lights up on the def
    //  it used. `sum-tok` = Σ over call tokens, which equals §27's (Σ Q)·K — a
    //  built-in check that the two agree. Raw dot, folded per tool (max over copies),
    //  ranked. Run `S28=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S28").is_ok() {
        fn dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        }
        println!("\n══ §28 — real per-token attention: call-token · def (peak vs summed) ══");
        let n_probes = tool_q_float.len();
        let phase_names = ["user", "user+think", "think+asst", "assistant"];
        println!(
            "  {} cases × 4 phases · rank vs {} tool defs · chance {:.1}% / {:.1}% (Top-1/Top-5)",
            n_probes,
            tool_set.len(),
            100.0 / tool_set.len() as f64,
            500.0 / tool_set.len() as f64,
        );
        println!(
            "  {:<14} {:>14} {:>14}",
            "phase", "max-tok T1/T5", "sum-tok T1/T5"
        );
        for p in 0..4 {
            let mut a_max = Acc::default();
            let mut a_sum = Acc::default();
            for i in 0..n_probes {
                let range = &tool_ranges[i][p];
                if range.is_empty() {
                    a_max.add(None);
                    a_sum.add(None);
                    continue;
                }
                let mut per_max: HashMap<&str, f32> = HashMap::new();
                let mut per_sum: HashMap<&str, f32> = HashMap::new();
                for (t, db) in def_mw.iter() {
                    let mut best = f32::MIN;
                    let mut sum = 0f32;
                    for &tk in range {
                        let s = dot(&tool_q_float[i][tk], db);
                        if s > best {
                            best = s;
                        }
                        sum += s;
                    }
                    let em = per_max.entry(t.as_str()).or_insert(f32::MIN);
                    *em = em.max(best);
                    let es = per_sum.entry(t.as_str()).or_insert(f32::MIN);
                    *es = es.max(sum);
                }
                a_max.add(rank_tool(&per_max, tool_phase_tool[i].as_str()));
                a_sum.add(rank_tool(&per_sum, tool_phase_tool[i].as_str()));
            }
            let pct = |a: &Acc| {
                (
                    100.0 * a.t1 as f64 / a.n.max(1) as f64,
                    100.0 * a.t5 as f64 / a.n.max(1) as f64,
                )
            };
            let (m1, m5) = pct(&a_max);
            let (s1, s5) = pct(&a_sum);
            println!(
                "  {:<14} {:>7.1}/{:>5.1} {:>7.1}/{:>5.1}",
                phase_names[p], m1, m5, s1, s5
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §29 — BASICS: true peak attention, per-token on BOTH sides.
    //
    //  score(def) = MAX over (call token, def token) of Q·K — the single strongest
    //  query→key link, the thing real attention actually maxes on. Def tokens are the
    //  first SWIN of the definition (tool name + signature). If even this peak doesn't
    //  rank the used def, no pooling is hiding a signal — the raw call→def link is
    //  genuinely absent and the rotation is required. Run `S29=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S29").is_ok() {
        fn dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        }
        println!("\n══ §29 — true peak attention: MAX over (call token × def token) of Q·K ══");
        let n_probes = tool_q_float.len();
        let phase_names = ["user", "user+think", "think+asst", "assistant"];
        println!(
            "  {} cases × 4 phases · rank vs {} tool defs · chance {:.1}% / {:.1}% (Top-1/Top-5)",
            n_probes,
            tool_set.len(),
            100.0 / tool_set.len() as f64,
            500.0 / tool_set.len() as f64,
        );
        println!("  {:<14} {:>12}", "phase", "peak T1/T5");
        for p in 0..4 {
            let mut acc = Acc::default();
            for i in 0..n_probes {
                let range = &tool_ranges[i][p];
                if range.is_empty() {
                    acc.add(None);
                    continue;
                }
                let mut per: HashMap<&str, f32> = HashMap::new();
                for (t, dtoks) in def_tok.iter() {
                    let mut best = f32::MIN;
                    for &tk in range {
                        let q = &tool_q_float[i][tk];
                        for dk in dtoks.iter() {
                            let s = dot(q, dk);
                            if s > best {
                                best = s;
                            }
                        }
                    }
                    let e = per.entry(t.as_str()).or_insert(f32::MIN);
                    *e = e.max(best);
                }
                acc.add(rank_tool(&per, tool_phase_tool[i].as_str()));
            }
            println!(
                "  {:<14} {:>6.1}/{:>5.1}",
                phase_names[p],
                100.0 * acc.t1 as f64 / acc.n.max(1) as f64,
                100.0 * acc.t5 as f64 / acc.n.max(1) as f64,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §30 — BASICS: single case, single token, raw dot-product statistics.
    //
    //  Pick one tool-call case. For every token in the tool-use (resp) span, take
    //  that one token's Q and dot it against every definition (mean-pooled, and the
    //  per-token PEAK). Report where the CORRECT def lands: its z-score and rank.
    //
    //  THIS IS THE DIAGNOSTIC THAT EXPOSED WHY §21–§29 READ AT CHANCE: the def
    //  candidate list (`def_mw` / `def_sign`) was contaminated with the system-prompt
    //  sections that the projection injects alongside the tool def (frame,
    //  reasoning_stance, grounding, …). `frame` is the attention SINK (positions 0–3,
    //  the largest-magnitude block in the whole context) and outscored every real tool
    //  ~8700 vs ~1600 — so the "winner" was the system frame on essentially every
    //  probe. We were dot-producting the call against the system scaffold, not the
    //  definitions. Fixed at the def-enumeration filter (~L688): exclude the named
    //  system sections so only real tool defs are candidates. `S30=1` (`S30_CASE`
    //  picks the case index, default 0). See docs §21.2.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S30").is_ok() {
        fn dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        }
        let i: usize = std::env::var("S30_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let tool = tool_phase_tool[i].clone();
        let resp = &tool_ranges[i][3];
        println!(
            "\n══ §30 — case {i} · tool={tool} · {} tool-use tokens · {} defs ══",
            resp.len(),
            def_mw.len()
        );
        println!("  for each tool-use token: where the CORRECT def's Q·K lands vs the other defs");
        println!(
            "  {:>5}  {:>7} {:>7}   {:>7} {:>7}",
            "tok", "mean-z", "mean-rk", "peak-z", "peak-rk"
        );
        for &tk in resp.iter() {
            let q = &tool_q_float[i][tk];
            let mvals: Vec<f32> = def_mw.iter().map(|(_, db)| dot(q, db)).collect();
            let pvals: Vec<f32> = def_tok
                .iter()
                .map(|(_, dt)| dt.iter().map(|dk| dot(q, dk)).fold(f32::MIN, f32::max))
                .collect();
            let cm = def_mw
                .iter()
                .zip(&mvals)
                .filter(|((t, _), _)| *t == tool)
                .map(|(_, &v)| v)
                .fold(f32::MIN, f32::max);
            let cp = def_tok
                .iter()
                .zip(&pvals)
                .filter(|((t, _), _)| *t == tool)
                .map(|(_, &v)| v)
                .fold(f32::MIN, f32::max);
            let (mm, ms) = mean_std(&mvals);
            let (pm, ps) = mean_std(&pvals);
            let mz = (cm - mm) / ms.max(1e-6);
            let pz = (cp - pm) / ps.max(1e-6);
            let mr = 1 + mvals.iter().filter(|&&v| v > cm).count();
            let pr = 1 + pvals.iter().filter(|&&v| v > cp).count();
            println!("  {tk:>5}  {mz:>7.2} {mr:>7}   {pz:>7.2} {pr:>7}");
        }
        // Full picture for the middle tool-use token: top-10 defs by peak dot.
        if !resp.is_empty() {
            let tk = resp[resp.len() / 2];
            let q = &tool_q_float[i][tk];
            let mut rows: Vec<(&str, f32)> = def_tok
                .iter()
                .map(|(t, dt)| {
                    (
                        t.as_str(),
                        dt.iter().map(|dk| dot(q, dk)).fold(f32::MIN, f32::max),
                    )
                })
                .collect();
            rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("  ── middle token {tk}: top-10 defs by peak Q·K ──");
            for (t, v) in rows.iter().take(10) {
                println!(
                    "    {v:>8.2}  {t}{}",
                    if *t == tool { "   ← CORRECT" } else { "" }
                );
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §31 — weighted sign-XOR-pop formula scan: 1 token → def, frozen 2026-06-27.
    //
    //  The rebuild after §30 cleaned the def list. Fix ONE call token (mid-tool-call)
    //  and score the correct def against all 93 by SEPARATION z, to make the mechanism
    //  legible before trusting any aggregate. Three pieces:
    //
    //  (1) ATOM — per head, the WEIGHTED sign disagreement against ONLY the def's own
    //      tokens: score(head) = Σ_d w[d]·[sign(Q[d]) ≠ sign(K[d])] over the head's 128
    //      dims. w[d]=1 recovers the plain 128-bit XOR-popcount (the production BDP atom).
    //  (2) WEIGHT — w[d] is a bounded integer from the importance |Q[d]| (the query
    //      magnitude IS how much a key dim matters to Q·K; it also kills the random
    //      sign-noise of near-zero-|Q| dims). Normalised within a group (head's 128 or
    //      layer's 512) and capped to [0,8] so the roll-up can never overflow. 13 weights
    //      = {uniform, rank, top, lin, sq, sqrt, log} × {head-, layer-(L:)normalised}.
    //  (3) ROLL-UP — def-tokens→head, heads→layer, layers→def, each reduced by one of 16
    //      formulas (sum, mean, p10…p100). Scan all 13×16³ ≈ 53k combos, rank the correct
    //      def by z, parallel over the combo space.
    //
    //  FROZEN RESULT (case 0 = telnet_session_list, tok 75, 93 defs): uniform 4.4σ →
    //  head-norm sq 6.8σ → LAYER-norm sq 9.6σ (rank 1). Levers, compounding: a high-tail
    //  roll-up (never sum/sum/sum) + the magnitude weight (steep `sq`) + LAYER
    //  normalisation (loud heads carry the signal; per-head norm flattens it).
    //  CAVEAT: one token, best-of-53k (noise floor ≈ 4.7σ) — 9.6σ is real on THIS token
    //  but NOT an accuracy; the cross-case k-fold (pick combo on train, score held-out)
    //  is the unrun generalisation test. Run `S21_ONLY=1 S31=1`. See docs §21.3–§21.5.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S31").is_ok() {
        use rayon::prelude::*;
        // Roll-up formula family: percentiles (fine grid, dense near the top end where
        // the winners clustered) plus sum/mean. `Pct(1.0)` = max, `Pct(0.0)` = min.
        #[derive(Clone, Copy)]
        enum F {
            Pct(f32),
            Sum,
            Mean,
        }
        impl F {
            fn apply(self, xs: &[f32]) -> f32 {
                match self {
                    F::Sum => xs.iter().sum(),
                    F::Mean => xs.iter().sum::<f32>() / xs.len().max(1) as f32,
                    F::Pct(p) => {
                        let mut v = xs.to_vec();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let idx = (((v.len().max(1) - 1) as f32) * p).round() as usize;
                        v[idx.min(v.len().saturating_sub(1))]
                    }
                }
            }
            fn label(self) -> String {
                match self {
                    F::Sum => "sum".to_string(),
                    F::Mean => "mean".to_string(),
                    F::Pct(p) => format!("p{:.0}", p * 100.0),
                }
            }
        }
        let grid: Vec<F> = {
            let mut g = vec![F::Sum, F::Mean];
            for &p in &[
                0.10f32, 0.25, 0.50, 0.65, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
                1.00,
            ] {
                g.push(F::Pct(p));
            }
            g
        };
        let ng = grid.len();
        const LW: usize = PER_LAYER_DIM / 64; // 8 words/layer
        const HW: usize = HEAD_DIM / 64; // 2 words/head
        let n_layers = BAND_HI - BAND_LO; // 16
        let n_heads = n_layers * N_KV_HEAD; // 64

        // One case, one probe token (mid tool-use), sign-packed.
        let i: usize = std::env::var("S31_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S31_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].as_str();
        let q_sign = sign_pack(&tool_q_float[i][tk]);

        // 4th scan axis: per-head_dim integer weight w[d] ∈ [0,8] from the importance
        // |Q[d]|, via a family of formulas, under TWO normalizations — within-head and
        // within-layer (L:) — so the scan picks which grouping sharpens the signal.
        const WN: usize = 13;
        let wlabel = [
            "uniform", "rank", "top", "lin", "sq", "sqrt", "log", "L:rank", "L:top", "L:lin",
            "L:sq", "L:sqrt", "L:log",
        ];
        let qmag: Vec<f32> = tool_q_float[i][tk].iter().map(|x| x.abs()).collect();
        // Fill the 6 formula slots wv[base..base+6] for one normalization group (the global
        // dim indices `group` — a head's 128 or a layer's 512), normalized over that group.
        fn fill(wv: &mut [Vec<u32>], base: usize, group: &[usize], qmag: &[f32]) {
            let mags: Vec<f32> = group.iter().map(|&d| qmag[d]).collect();
            let mh = mags.iter().cloned().fold(f32::MIN, f32::max).max(1e-9);
            let mut order: Vec<usize> = (0..group.len()).collect();
            order.sort_by(|&a, &b| mags[a].partial_cmp(&mags[b]).unwrap());
            let mut rank = vec![0usize; group.len()];
            for (r, &j) in order.iter().enumerate() {
                rank[j] = r;
            }
            let n = group.len();
            let topk = (n / 8).max(1); // top 12.5%: 16 of 128 (head), 64 of 512 (layer)
            for (j, &d) in group.iter().enumerate() {
                let m = mags[j];
                let r = m / mh;
                wv[base][d] = ((8.0 * rank[j] as f32) / (n as f32 - 1.0)).round() as u32;
                wv[base + 1][d] = if rank[j] >= n - topk { 8 } else { 1 };
                wv[base + 2][d] = (8.0 * r).round() as u32;
                wv[base + 3][d] = (8.0 * r * r).round() as u32;
                wv[base + 4][d] = (8.0 * r.sqrt()).round() as u32;
                wv[base + 5][d] = (8.0 * (1.0 + m).ln() / (1.0 + mh).ln().max(1e-9)).round() as u32;
            }
        }
        let weight_vecs: Vec<Vec<u32>> = {
            let mut wv = vec![vec![1u32; qmag.len()]; WN]; // wv[0] = uniform
            for hh in 0..n_heads {
                let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                let group: Vec<usize> = (base..base + HEAD_DIM).collect();
                fill(&mut wv, 1, &group, &qmag); // head-normalized → wv[1..7]
            }
            for lp in 0..n_layers {
                let base = lp * PER_LAYER_DIM;
                let group: Vec<usize> = (base..base + PER_LAYER_DIM).collect();
                fill(&mut wv, 7, &group, &qmag); // layer-normalized → wv[7..13]
            }
            wv
        };
        fn wmis(x0: u64, x1: u64, wts: &[u32]) -> u32 {
            let mut s = 0u32;
            let mut m = x0;
            while m != 0 {
                s += wts[m.trailing_zeros() as usize];
                m &= m - 1;
            }
            let mut m = x1;
            while m != 0 {
                s += wts[64 + m.trailing_zeros() as usize];
                m &= m - 1;
            }
            s
        }

        println!(
            "\n══ §31 — weighted sign-XOR formula scan: 1 token → def, weight × 3-level roll-up ══"
        );
        println!(
            "  case {i} · tool={truth} · probe token {tk} · {} defs · atom = Σ_d w[d]·[sign Q ≠ sign K] per head",
            def_sign.len()
        );
        println!("  weight w[d] from |Q[d]| ∈ {wlabel:?}");

        // Per def: head_f1[head][weight][f1] — F1 reductions (over def tokens) of the
        // per-head weighted mismatch, for each weight formula. Parallel over defs.
        let per_def: Vec<Vec<Vec<Vec<f32>>>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                let mut sc: Vec<Vec<Vec<f32>>> =
                    vec![vec![Vec::with_capacity(toks.len()); WN]; n_heads];
                for tw in toks {
                    for hh in 0..n_heads {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let x0 = q_sign[wb] ^ tw[wb];
                        let x1 = q_sign[wb + 1] ^ tw[wb + 1];
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        for w in 0..WN {
                            sc[hh][w]
                                .push(wmis(x0, x1, &weight_vecs[w][base..base + HEAD_DIM]) as f32);
                        }
                    }
                }
                (0..n_heads)
                    .map(|hh| {
                        (0..WN)
                            .map(|w| {
                                let mut s = sc[hh][w].clone();
                                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                let sum: f32 = s.iter().sum();
                                let n = s.len().max(1) as f32;
                                grid.iter()
                                    .map(|f| match f {
                                        F::Sum => sum,
                                        F::Mean => sum / n,
                                        F::Pct(p) => {
                                            s[((((s.len().max(1) - 1) as f32) * p).round()
                                                as usize)
                                                .min(s.len().saturating_sub(1))]
                                        }
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect()
            })
            .collect();

        // Scan F1 × F2 × F3 → a score per def; rank the correct def by separation (z).
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| t == truth)
            .map(|(j, _)| j)
            .collect();
        // Scan weight × head × layer × def (parallel over the flattened combo space).
        let total = WN * ng * ng * ng;
        let mut rows: Vec<(f64, f64, usize, usize, usize, usize, usize)> = (0..total)
            .into_par_iter()
            .map(|ci| {
                let c = ci % ng;
                let b = (ci / ng) % ng;
                let a = (ci / (ng * ng)) % ng;
                let w = ci / (ng * ng * ng);
                let scores: Vec<f32> = per_def
                    .iter()
                    .map(|hf| {
                        let layers: Vec<f32> = (0..n_layers)
                            .map(|l| {
                                let heads: [f32; N_KV_HEAD] =
                                    std::array::from_fn(|h| hf[l * N_KV_HEAD + h][w][a]);
                                grid[b].apply(&heads)
                            })
                            .collect();
                        grid[c].apply(&layers)
                    })
                    .collect();
                let (mean, std) = mean_std(&scores);
                let mut bz = 0f64;
                let mut brank = scores.len();
                for &td in &truth_defs {
                    let z = (scores[td] - mean) as f64 / std.max(1e-6) as f64;
                    let rank = if z < 0.0 {
                        1 + scores.iter().filter(|&&v| v < scores[td]).count()
                    } else {
                        1 + scores.iter().filter(|&&v| v > scores[td]).count()
                    };
                    if z.abs() > bz.abs() {
                        bz = z;
                        brank = rank;
                    }
                }
                (bz.abs(), bz, brank, w, a, b, c)
            })
            .collect();
        rows.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        println!(
            "  top combos by |z| of the correct def among {} defs:",
            def_sign.len()
        );
        println!(
            "  {:>7} {:>6}   {:<7} {:<5} {:<5} {:<5}",
            "z", "rank", "weight", "head", "layer", "def"
        );
        for (_, z, rank, w, a, b, c) in rows.iter().take(20) {
            println!(
                "  {:>7.2} {:>6}   {:<7} {:<5} {:<5} {:<5}",
                z,
                rank,
                wlabel[*w],
                grid[*a].label(),
                grid[*b].label(),
                grid[*c].label()
            );
        }
        if let Some(r) = rows
            .iter()
            .find(|(_, _, _, w, a, b, c)| *w == 0 && *a == 0 && *b == 0 && *c == 0)
        {
            println!(
                "  baseline uniform sum/sum/sum:  z={:.2}  rank={}",
                r.1, r.2
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §32 — LOCKED formula (frozen §31, 2026-06-27): the positive test on one token.
    //
    //  §31's scan answered the formula question; §32 fixes that answer and DROPS the
    //  four scanned dimensions (weight/head/layer/def formulas), leaving a clean
    //  single-score test we extend with NEW dimensions next. Locked:
    //    weight = uniform (w[d]=1, the plain sign-XOR-popcount atom),
    //    head   = p25     (percentile over the def's tokens → per-head value),
    //    layer  = p75     (percentile over the 4 heads      → per-layer value),
    //    def    = p100    (max over the 16 layers           → def score).
    //  We moved here from the §31 negative-z L:sq winner after a robustness probe: a
    //  one-notch def percentile shift (p10→p20) collapsed L:sq to chance, exposing it
    //  as a knife-edge fluke. This positive-z uniform formula is the sturdier candidate
    //  under test. The one probe token is scored against EVERY def (its target + all
    //  distractors); the correct def is ranked by separation z, direction chosen from
    //  the target's own z sign. Run `S21_ONLY=1 S32=1` (`S32_CASE`/`S32_TOK` pick the
    //  case/token). See docs §21.4.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S32").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S32_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S32_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].as_str();
        let q_sign = sign_pack(&tool_q_float[i][tk]);

        // Locked weight: uniform (w[d]=1) — the plain sign-XOR-popcount atom. The
        // negative-z L:sq winner was knife-edge (a one-notch percentile shift collapsed
        // it), so we move to the positive-z uniform formula to test for robustness.
        let w = vec![1u32; tool_q_float[i][tk].len()];
        fn wmis(x0: u64, x1: u64, wts: &[u32]) -> u32 {
            let mut s = 0u32;
            let mut m = x0;
            while m != 0 {
                s += wts[m.trailing_zeros() as usize];
                m &= m - 1;
            }
            let mut m = x1;
            while m != 0 {
                s += wts[64 + m.trailing_zeros() as usize];
                m &= m - 1;
            }
            s
        }
        fn pct(v: &mut [f32], p: f32) -> f32 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((((v.len().max(1) - 1) as f32) * p).round() as usize).min(v.len().saturating_sub(1))]
        }

        // Locked score per def: p100_layers( p75_heads( p25_deftokens( mismatch ) ) ).
        let scores: Vec<f32> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                let mut head_val = vec![0f32; n_heads];
                for hh in 0..n_heads {
                    let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                    let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                    let mut pops: Vec<f32> = toks
                        .iter()
                        .map(|tw| {
                            wmis(
                                q_sign[wb] ^ tw[wb],
                                q_sign[wb + 1] ^ tw[wb + 1],
                                &w[base..base + HEAD_DIM],
                            ) as f32
                        })
                        .collect();
                    head_val[hh] = pct(&mut pops, 0.25);
                }
                let mut layer_val: Vec<f32> = (0..n_layers)
                    .map(|l| {
                        let mut hs: Vec<f32> = (0..N_KV_HEAD)
                            .map(|h| head_val[l * N_KV_HEAD + h])
                            .collect();
                        pct(&mut hs, 0.75)
                    })
                    .collect();
                pct(&mut layer_val, 1.00)
            })
            .collect();

        // Positive test: rank the correct def among ALL defs (target + distractors).
        // This formula separates the target with POSITIVE z (correct def scores HIGH),
        // so higher = better; pick the direction from the target's own z sign.
        let (mean, std) = mean_std(&scores);
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| t == truth)
            .map(|(j, _)| j)
            .collect();
        let hi_is_better = truth_defs
            .iter()
            .map(|&td| (scores[td] - mean) / std.max(1e-6))
            .sum::<f32>()
            >= 0.0;
        let mut order: Vec<usize> = (0..scores.len()).collect();
        if hi_is_better {
            order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        } else {
            order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
        }

        println!("\n══ §32 — LOCKED formula (uniform · head p25 · layer p75 · def p100) — positive test ══");
        println!(
            "  case {i} · tool={truth} · probe token {tk} · 1 target + {} distractor defs",
            scores.len() - truth_defs.len()
        );
        println!(
            "  {} defs — target marked ◀:",
            if hi_is_better {
                "highest-scoring (best-matching)"
            } else {
                "lowest-scoring (best-matching)"
            }
        );
        for (rank, &d) in order.iter().take(8).enumerate() {
            let z = (scores[d] - mean) as f64 / std.max(1e-6) as f64;
            let mark = if truth_defs.contains(&d) {
                " ◀ TARGET"
            } else {
                ""
            };
            println!(
                "  {:>3}. {:<28} score={:>7.1}  z={:>6.2}{}",
                rank + 1,
                def_sign[d].0,
                scores[d],
                z,
                mark
            );
        }
        for &td in &truth_defs {
            let z = (scores[td] - mean) as f64 / std.max(1e-6) as f64;
            let rank = if hi_is_better {
                1 + scores.iter().filter(|&&v| v > scores[td]).count()
            } else {
                1 + scores.iter().filter(|&&v| v < scores[td]).count()
            };
            println!(
                "  target '{}': rank {}/{}  z={:.2}",
                def_sign[td].0,
                rank,
                scores.len(),
                z
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §33 — ROBUSTNESS sweep of the §32 uniform p25/p75/p100 formula.
    //
    //  The L:sq winner collapsed when one percentile moved a notch (def p10→p20). §33
    //  perturbs EACH of the three percentiles around the locked uniform point, holding
    //  the other two fixed, and reports the target's rank + z AND the same-family
    //  siblings' mean z (does the `*_session_list` cluster stay elevated?). Robust =
    //  holds rank 1 with the family clustered across the neighborhood; fluke = collapses
    //  to chance. `family` = defs sharing the target's last-two `_`-segments (its stem).
    //  Run `S21_ONLY=1 S33=1` (`S33_CASE`/`S33_TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S33").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S33_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S33_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);

        // Per-(def,head) sorted popcounts (uniform weight), precomputed once so the
        // percentile sweep is just index lookups.
        let sorted_pops: Vec<Vec<Vec<f32>>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p
                    })
                    .collect()
            })
            .collect();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        // (target rank, target z, family mean z) for one percentile triple.
        let eval = |hp: f32, lp: f32, dp: f32| -> (usize, f64, f64) {
            let scores: Vec<f32> = (0..def_sign.len())
                .map(|di| {
                    let head_val: Vec<f32> = (0..n_heads)
                        .map(|hh| {
                            let s = &sorted_pops[di][hh];
                            s[pidx(s.len(), hp)]
                        })
                        .collect();
                    let mut layer_val: Vec<f32> = (0..n_layers)
                        .map(|l| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, lp)]
                        })
                        .collect();
                    layer_val.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    layer_val[pidx(n_layers, dp)]
                })
                .collect();
            let (mean, std) = mean_std(&scores);
            let mut tz = 0f64;
            let mut trank = scores.len();
            for &td in &truth_defs {
                let z = (scores[td] - mean) as f64 / std.max(1e-6) as f64;
                let rank = if z >= 0.0 {
                    1 + scores.iter().filter(|&&v| v > scores[td]).count()
                } else {
                    1 + scores.iter().filter(|&&v| v < scores[td]).count()
                };
                if z.abs() > tz.abs() {
                    tz = z;
                    trank = rank;
                }
            }
            let fam_z = if family.is_empty() {
                0.0
            } else {
                family
                    .iter()
                    .map(|&f| (scores[f] - mean) as f64 / std.max(1e-6) as f64)
                    .sum::<f64>()
                    / family.len() as f64
            };
            (trank, tz, fam_z)
        };

        println!(
            "\n══ §33 — robustness sweep around uniform p25/p75/p100 (case {i}, tool={truth}) ══"
        );
        println!(
            "  family = {} sibling defs sharing stem '{}'  ·  {} defs, chance rank ~{}",
            family.len(),
            tstem,
            def_sign.len(),
            def_sign.len() / 2
        );
        let row = |label: String, r: usize, z: f64, fz: f64| {
            println!(
                "    {:<6} target rank {:>3}  z={:>6.2}   family ⟨z⟩={:>5.2}",
                label, r, z, fz
            );
        };
        println!("  axis = HEAD percentile (layer p75, def p100):");
        for &hp in &[0.10f32, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] {
            let (r, z, fz) = eval(hp, 0.75, 1.00);
            row(format!("p{:.0}", hp * 100.0), r, z, fz);
        }
        println!("  axis = LAYER percentile (head p25, def p100):");
        for &lp in &[0.55f32, 0.65, 0.70, 0.75, 0.80, 0.85, 0.95] {
            let (r, z, fz) = eval(0.25, lp, 1.00);
            row(format!("p{:.0}", lp * 100.0), r, z, fz);
        }
        println!("  axis = DEF percentile (head p25, layer p75):");
        for &dp in &[0.80f32, 0.85, 0.90, 0.94, 0.96, 0.98, 1.00] {
            let (r, z, fz) = eval(0.25, 0.75, dp);
            row(format!("p{:.0}", dp * 100.0), r, z, fz);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §34 — per-LAYER differentiation, head LOCKED at p25.
    //
    //  §33 showed the def-axis signal collapses to one max-over-layers layer. So rank
    //  the 16 routing-band layers (L24–L39) by how strongly EACH ONE ALONE separates
    //  the target from the field: with head locked p25 and the head→layer roll-up p75,
    //  each layer gives one score per def; report the target's z + rank and the family
    //  ⟨z⟩ at that layer, sorted by the target's differentiating z. This says which
    //  layers carry the routing signal (keep) vs. which are dead weight (drop), so we
    //  can replace the fragile max-over-all-16 with a combine over the good layers.
    //  Run `S21_ONLY=1 S34=1` (`S34_CASE`/`S34_TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S34").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S34_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S34_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25; // LOCKED
        const LAYER_P: f32 = 0.75; // head→layer roll-up (kept)

        // head_val[def][head] = p25 over the def's tokens of the uniform mismatch.
        let head_val: Vec<Vec<f32>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p[pidx(p.len(), HEAD_P)]
                    })
                    .collect()
            })
            .collect();

        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        // Per layer: one score per def = p75 over its 4 heads; rank the target.
        let mut rows: Vec<(f64, usize, usize, f64)> = Vec::new(); // (target z, layer, rank, family z)
        for l in 0..n_layers {
            let scores: Vec<f32> = (0..def_sign.len())
                .map(|di| {
                    let mut hs: Vec<f32> = (0..N_KV_HEAD)
                        .map(|h| head_val[di][l * N_KV_HEAD + h])
                        .collect();
                    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    hs[pidx(N_KV_HEAD, LAYER_P)]
                })
                .collect();
            let (mean, std) = mean_std(&scores);
            let mut tz = 0f64;
            let mut trank = scores.len();
            for &td in &truth_defs {
                let z = (scores[td] - mean) as f64 / std.max(1e-6) as f64;
                let rank = if z >= 0.0 {
                    1 + scores.iter().filter(|&&v| v > scores[td]).count()
                } else {
                    1 + scores.iter().filter(|&&v| v < scores[td]).count()
                };
                if z.abs() > tz.abs() {
                    tz = z;
                    trank = rank;
                }
            }
            let fam_z = if family.is_empty() {
                0.0
            } else {
                family
                    .iter()
                    .map(|&f| (scores[f] - mean) as f64 / std.max(1e-6) as f64)
                    .sum::<f64>()
                    / family.len() as f64
            };
            rows.push((tz, l, trank, fam_z));
        }
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // most-differentiating first

        println!("\n══ §34 — layers ranked by differentiating score (case {i}, tool={truth}) ══");
        println!(
            "  head LOCKED p25 · head→layer p75 · {} defs · family = {} siblings (stem '{}')",
            def_sign.len(),
            family.len(),
            tstem
        );
        println!(
            "  {:>5}   {:>7}   {:>9}   {:>9}",
            "layer", "tgt z", "tgt rank", "fam ⟨z⟩"
        );
        for (tz, l, trank, fz) in &rows {
            println!(
                "   L{:<3}   {:>7.2}   {:>5}/{:<3}   {:>9.2}",
                BAND_LO + l,
                tz,
                trank,
                def_sign.len(),
                fz
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §35 — sum the target-positive layers (head LOCKED p25, head→layer p75).
    //
    //  §34 ranks each layer by how well it differentiates the target. §35 SELECTS the
    //  layers where the target scores positively (it stands out HIGH there), takes the
    //  top-K by differentiating z, and SUMS their per-def scores into one combined
    //  score — replacing the fragile max-over-all-layers with a consensus over the good
    //  layers. Then the positive test: rank the target by the combined score.
    //  CAVEAT: layers are selected using the target's own z (in-sample for this token);
    //  the honest version selects layers on TRAIN cases and scores held-out. `S35_K`
    //  caps the layer count (default 8). Run `S21_ONLY=1 S35=1` (`S35_CASE`/`S35_TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S35").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S35_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S35_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let kk: usize = std::env::var("S35_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;

        // head_val[def][head] = p25 over the def's tokens of the uniform mismatch.
        let head_val: Vec<Vec<f32>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p[pidx(p.len(), HEAD_P)]
                    })
                    .collect()
            })
            .collect();

        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        // Per-layer score per def, and the target's z at each layer.
        let layer_scores: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..def_sign.len())
                    .map(|di| {
                        let mut hs: Vec<f32> = (0..N_KV_HEAD)
                            .map(|h| head_val[di][l * N_KV_HEAD + h])
                            .collect();
                        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        hs[pidx(N_KV_HEAD, LAYER_P)]
                    })
                    .collect()
            })
            .collect();
        let mut lz: Vec<(f64, usize)> = (0..n_layers)
            .map(|l| {
                let (mean, std) = mean_std(&layer_scores[l]);
                let mut tz = 0f64;
                for &td in &truth_defs {
                    let z = (layer_scores[l][td] - mean) as f64 / std.max(1e-6) as f64;
                    if z.abs() > tz.abs() {
                        tz = z;
                    }
                }
                (tz, l)
            })
            .collect();
        lz.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let selected: Vec<usize> = lz
            .iter()
            .filter(|(z, _)| *z > 0.0)
            .take(kk)
            .map(|(_, l)| *l)
            .collect();

        // Combined score = SUM over the selected (target-positive) layers.
        let combined: Vec<f32> = (0..def_sign.len())
            .map(|di| selected.iter().map(|&l| layer_scores[l][di]).sum())
            .collect();
        let (mean, std) = mean_std(&combined);
        let mut order: Vec<usize> = (0..combined.len()).collect();
        order.sort_by(|&a, &b| combined[b].partial_cmp(&combined[a]).unwrap()); // hi = better

        println!("\n══ §35 — sum of target-positive layers (case {i}, tool={truth}) ══");
        let sel_lbl: Vec<String> = selected
            .iter()
            .map(|&l| format!("L{}", BAND_LO + l))
            .collect();
        println!(
            "  selected {} layers (target z>0, top-{}): {}",
            selected.len(),
            kk,
            sel_lbl.join(" ")
        );
        println!("  highest-scoring defs (combined) — target ◀:");
        for (rank, &d) in order.iter().take(8).enumerate() {
            let z = (combined[d] - mean) as f64 / std.max(1e-6) as f64;
            let mark = if truth_defs.contains(&d) {
                " ◀ TARGET"
            } else {
                ""
            };
            println!(
                "  {:>3}. {:<28} score={:>7.1}  z={:>6.2}{}",
                rank + 1,
                def_sign[d].0,
                combined[d],
                z,
                mark
            );
        }
        let fam_z = if family.is_empty() {
            0.0
        } else {
            family
                .iter()
                .map(|&f| (combined[f] - mean) as f64 / std.max(1e-6) as f64)
                .sum::<f64>()
                / family.len() as f64
        };
        for &td in &truth_defs {
            let z = (combined[td] - mean) as f64 / std.max(1e-6) as f64;
            let rank = 1 + combined.iter().filter(|&&v| v > combined[td]).count();
            println!(
                "  target '{}': rank {}/{}  z={:.2}   family ⟨z⟩={:.2}",
                def_sign[td].0,
                rank,
                combined.len(),
                z,
                fam_z
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §36 — signed-|z| layer combine (head LOCKED p25, head→layer p75).
    //
    //  §34 showed the layers split on direction: some single out the target by HIGH
    //  mismatch (+z, e.g. L32), others by LOW mismatch (−z, e.g. L34). §35 summed only
    //  the +z half and threw the −z half away. §36 ranks layers by |target z| (strength
    //  in EITHER direction), and combines each layer's per-def z folded by the layer's
    //  target-direction sign: combined = Σ_l sign(tz_l)·z_l[def]. The target then
    //  accumulates |tz_l| from every layer (both directions reinforce); distractors get
    //  random-signed z that cancels. Swept over layer count K.
    //  CAVEAT: layers AND their signs are picked from the target's own z — maximally
    //  in-sample for this token; this is the CEILING of layer-combination, the honest
    //  number needs train/holdout layer selection. Run `S21_ONLY=1 S36=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S36").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S36_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S36_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;

        let head_val: Vec<Vec<f32>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p[pidx(p.len(), HEAD_P)]
                    })
                    .collect()
            })
            .collect();

        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        // Per-layer score per def, and per-layer (mean, std, target z).
        let layer_scores: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..def_sign.len())
                    .map(|di| {
                        let mut hs: Vec<f32> = (0..N_KV_HEAD)
                            .map(|h| head_val[di][l * N_KV_HEAD + h])
                            .collect();
                        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        hs[pidx(N_KV_HEAD, LAYER_P)]
                    })
                    .collect()
            })
            .collect();
        let layer_stats: Vec<(f32, f32, f64)> = (0..n_layers)
            .map(|l| {
                let (mean, std) = mean_std(&layer_scores[l]);
                let mut tz = 0f64;
                for &td in &truth_defs {
                    let z = (layer_scores[l][td] - mean) as f64 / std.max(1e-6) as f64;
                    if z.abs() > tz.abs() {
                        tz = z;
                    }
                }
                (mean, std, tz)
            })
            .collect();
        // Layers ranked by |target z| (strength in either direction).
        let mut by_abs: Vec<usize> = (0..n_layers).collect();
        by_abs.sort_by(|&a, &b| {
            layer_stats[b]
                .2
                .abs()
                .partial_cmp(&layer_stats[a].2.abs())
                .unwrap()
        });

        // Signed-|z| combine over the top-k layers.
        let eval_k = |k: usize| -> (usize, f64, f64) {
            let sel = &by_abs[..k.min(n_layers)];
            let combined: Vec<f32> = (0..def_sign.len())
                .map(|di| {
                    sel.iter()
                        .map(|&l| {
                            let (mean, std, tz) = layer_stats[l];
                            let s = if tz >= 0.0 { 1.0 } else { -1.0 };
                            s * (layer_scores[l][di] - mean) / std.max(1e-6)
                        })
                        .sum()
                })
                .collect();
            let (mean, std) = mean_std(&combined);
            let mut tz = 0f64;
            let mut trank = combined.len();
            for &td in &truth_defs {
                let z = (combined[td] - mean) as f64 / std.max(1e-6) as f64;
                let rank = 1 + combined.iter().filter(|&&v| v > combined[td]).count();
                if z.abs() > tz.abs() {
                    tz = z;
                    trank = rank;
                }
            }
            let fam_z = if family.is_empty() {
                0.0
            } else {
                family
                    .iter()
                    .map(|&f| (combined[f] - mean) as f64 / std.max(1e-6) as f64)
                    .sum::<f64>()
                    / family.len() as f64
            };
            (trank, tz, fam_z)
        };

        println!("\n══ §36 — signed-|z| layer combine (case {i}, tool={truth}) ══");
        let top12: Vec<String> = by_abs
            .iter()
            .take(12)
            .map(|&l| {
                format!(
                    "L{}{}",
                    BAND_LO + l,
                    if layer_stats[l].2 >= 0.0 { "+" } else { "-" }
                )
            })
            .collect();
        println!(
            "  layers by |target z| (sign = direction): {}",
            top12.join(" ")
        );
        println!(
            "  {:>4}   {:>9}   {:>7}   {:>9}",
            "K", "tgt rank", "tgt z", "fam ⟨z⟩"
        );
        for &k in &[1usize, 2, 4, 8, 12, 16, 24, 48] {
            let (r, z, fz) = eval_k(k);
            println!(
                "  {:>4}   {:>5}/{:<3}   {:>7.2}   {:>9.2}",
                k,
                r,
                def_sign.len(),
                z,
                fz
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §37 — does the differentiating layer set change as the DECODE TOKEN changes?
    //
    //  §31–§36 all locked one token (mid tool-call). §37 adds the token dimension:
    //  for EVERY token in the tool-call span, recompute the per-layer target z (head
    //  LOCKED p25, head→layer p75) and the §36 signed-|z| K=4 combine rank. Then a
    //  LAYER-CONSENSUS: per layer, its mean signed z across tokens and how many tokens
    //  rank it a top-6 differentiator. If L32+/L34− recur across tokens, the layer set
    //  is a property of the CALL (stable, calibratable); if it scatters, it is per-token
    //  noise. Run `S21_ONLY=1 S37=1` (`S37_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S37").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S37_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let n_defs = def_sign.len();

        // Per token: (token pos, per-layer signed target z, K=4 signed-combine rank/z).
        let per_token: Vec<(usize, Vec<f64>, usize, f64)> = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                // head_val[def][head] = p25 over def tokens of the uniform mismatch.
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                let layer_scores: Vec<Vec<f32>> = (0..n_layers)
                    .map(|l| {
                        (0..n_defs)
                            .map(|di| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[di][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect()
                    })
                    .collect();
                let stats: Vec<(f32, f32, f64)> = (0..n_layers)
                    .map(|l| {
                        let (mean, std) = mean_std(&layer_scores[l]);
                        let mut tz = 0f64;
                        for &td in &truth_defs {
                            let z = (layer_scores[l][td] - mean) as f64 / std.max(1e-6) as f64;
                            if z.abs() > tz.abs() {
                                tz = z;
                            }
                        }
                        (mean, std, tz)
                    })
                    .collect();
                let layer_z: Vec<f64> = stats.iter().map(|s| s.2).collect();
                // K=4 signed-|z| combine for this token.
                let mut by_abs: Vec<usize> = (0..n_layers).collect();
                by_abs.sort_by(|&a, &b| layer_z[b].abs().partial_cmp(&layer_z[a].abs()).unwrap());
                let sel = &by_abs[..4.min(n_layers)];
                let combined: Vec<f32> = (0..n_defs)
                    .map(|di| {
                        sel.iter()
                            .map(|&l| {
                                let (mean, std, tz) = stats[l];
                                let s = if tz >= 0.0 { 1.0 } else { -1.0 };
                                s * (layer_scores[l][di] - mean) / std.max(1e-6)
                            })
                            .sum()
                    })
                    .collect();
                let (cm, cs) = mean_std(&combined);
                let mut cz = 0f64;
                let mut crank = n_defs;
                for &td in &truth_defs {
                    let z = (combined[td] - cm) as f64 / cs.max(1e-6) as f64;
                    let rank = 1 + combined.iter().filter(|&&v| v > combined[td]).count();
                    if z.abs() > cz.abs() {
                        cz = z;
                        crank = rank;
                    }
                }
                (tk, layer_z, crank, cz)
            })
            .collect();

        println!(
            "\n══ §37 — layer set vs decode token (case {i}, tool={truth}, {} tokens) ══",
            resp.len()
        );
        println!("  per token: top-4 layers by |z| (signed) · K=4 signed-combine rank/z:");
        for (tk, lz, crank, cz) in per_token.iter() {
            let mut by_abs: Vec<usize> = (0..n_layers).collect();
            by_abs.sort_by(|&a, &b| lz[b].abs().partial_cmp(&lz[a].abs()).unwrap());
            let top: Vec<String> = by_abs
                .iter()
                .take(4)
                .map(|&l| format!("L{}{}", BAND_LO + l, if lz[l] >= 0.0 { "+" } else { "-" }))
                .collect();
            println!(
                "    tok {:>3}:  {:<24}  rank {:>3}/{}  z={:>5.2}",
                tk,
                top.join(" "),
                crank,
                n_defs,
                cz
            );
        }

        // Layer consensus across tokens: mean signed z + top-6 recurrence count.
        let nt = per_token.len().max(1);
        let mut cons: Vec<(usize, f64, usize)> = (0..n_layers)
            .map(|l| {
                let mean_z: f64 =
                    per_token.iter().map(|(_, lz, _, _)| lz[l]).sum::<f64>() / nt as f64;
                let top6 = per_token
                    .iter()
                    .filter(|(_, lz, _, _)| {
                        let mut idx: Vec<usize> = (0..n_layers).collect();
                        idx.sort_by(|&a, &b| lz[b].abs().partial_cmp(&lz[a].abs()).unwrap());
                        idx[..6.min(n_layers)].contains(&l)
                    })
                    .count();
                (l, mean_z, top6)
            })
            .collect();
        cons.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then(b.1.abs().partial_cmp(&a.1.abs()).unwrap())
        });
        println!("  LAYER CONSENSUS across {nt} tokens (sorted by top-6 recurrence):");
        println!(
            "    {:>5}   {:>8}   {:>11}",
            "layer", "mean z", "top-6 in N"
        );
        for (l, mz, t6) in cons.iter().take(14) {
            println!(
                "     L{:<3}   {:>8.2}   {:>4}/{:<4}",
                BAND_LO + l,
                mz,
                t6,
                nt
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §38 — can a BLIND (label-free) score property select the best layer?
    //
    //  §37 showed the best layer changes per token, and selecting it by target z is
    //  in-sample cheating. §38 asks the holdout question: is there a property of a
    //  layer's score distribution over ALL 93 candidate defs — computable WITHOUT
    //  knowing which is correct — that CORRELATES with the layer's true discriminative
    //  power (target |z|)? For each (token × layer) we compute the oracle quality
    //  (target |z|) and several blind shape props of the 93-def z-distribution
    //  (raw std, max|z|, isolation-margin of the most extreme def, kurtosis, range).
    //  Report (a) the per-token correlation of each prop with target |z|, and (b) how
    //  much of the oracle's best-layer quality each prop RECOVERS when used to pick the
    //  layer blind. A high-correlation prop is the holdout layer-selector. Run
    //  `S21_ONLY=1 S38=1` (`S38_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S38").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S38_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn pearson(x: &[f64], y: &[f64]) -> f64 {
            let n = x.len() as f64;
            let mx = x.iter().sum::<f64>() / n;
            let my = y.iter().sum::<f64>() / n;
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for k in 0..x.len() {
                let (dx, dy) = (x[k] - mx, y[k] - my);
                sxy += dx * dy;
                sxx += dx * dx;
                syy += dy * dy;
            }
            if sxx <= 0.0 || syy <= 0.0 {
                0.0
            } else {
                sxy / (sxx.sqrt() * syy.sqrt())
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        const NPROP: usize = 5;
        let pnames = ["raw_std", "max|z|", "margin", "kurtosis", "range"];

        // per_token[t][layer] = (target |z|, [5 blind props]).
        let per_token: Vec<Vec<(f64, [f64; NPROP])>> = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                (0..n_layers)
                    .map(|l| {
                        let scores: Vec<f32> = (0..n_defs)
                            .map(|di| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[di][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect();
                        let (mean, std) = mean_std(&scores);
                        let zs: Vec<f64> = scores
                            .iter()
                            .map(|&s| (s - mean) as f64 / std.max(1e-6) as f64)
                            .collect();
                        let mut tz = 0f64;
                        for &td in &truth_defs {
                            if zs[td].abs() > tz {
                                tz = zs[td].abs();
                            }
                        }
                        let mut absz: Vec<f64> = zs.iter().map(|z| z.abs()).collect();
                        absz.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        let max_absz = absz[0];
                        let margin = absz[0] - absz.get(1).copied().unwrap_or(0.0);
                        let kurt = zs.iter().map(|z| z.powi(4)).sum::<f64>() / n_defs as f64 - 3.0;
                        let lo = zs.iter().cloned().fold(f64::MAX, f64::min);
                        let hi = zs.iter().cloned().fold(f64::MIN, f64::max);
                        (tz, [std as f64, max_absz, margin, kurt, hi - lo])
                    })
                    .collect()
            })
            .collect();

        // (a) per-token correlation of each prop with target |z|, averaged.
        let nt = per_token.len().max(1);
        let mut corr = [0f64; NPROP];
        for tokrows in &per_token {
            let y: Vec<f64> = tokrows.iter().map(|(tz, _)| *tz).collect();
            for (j, c) in corr.iter_mut().enumerate() {
                let x: Vec<f64> = tokrows.iter().map(|(_, p)| p[j]).collect();
                *c += pearson(&x, &y);
            }
        }
        for c in corr.iter_mut() {
            *c /= nt as f64;
        }
        // (b) recovery: pick the layer by each prop (blind) → target |z| there; vs oracle/mean.
        let mut recov = [0f64; NPROP];
        let (mut oracle, mut meanq) = (0f64, 0f64);
        for tokrows in &per_token {
            let argmax = |key: &dyn Fn(&(f64, [f64; NPROP])) -> f64| -> f64 {
                let l = (0..tokrows.len())
                    .max_by(|&a, &b| key(&tokrows[a]).partial_cmp(&key(&tokrows[b])).unwrap())
                    .unwrap();
                tokrows[l].0
            };
            for (j, r) in recov.iter_mut().enumerate() {
                *r += argmax(&|row| row.1[j]);
            }
            oracle += argmax(&|row| row.0);
            meanq += tokrows.iter().map(|(tz, _)| *tz).sum::<f64>() / tokrows.len() as f64;
        }

        println!("\n══ §38 — blind score-property → best-layer selection (case {i}, tool={truth}, {nt} tokens) ══");
        println!("  (a) mean per-token Pearson r ( blind prop  vs  layer's target |z| ):");
        for j in 0..NPROP {
            println!("       {:<9} r = {:>6.3}", pnames[j], corr[j]);
        }
        println!("  (b) target |z| recovered by picking the layer BLIND via each prop:");
        println!("       {:<9} {:>6.2}", "ORACLE", oracle / nt as f64);
        for j in 0..NPROP {
            println!("       {:<9} {:>6.2}", pnames[j], recov[j] / nt as f64);
        }
        println!(
            "       {:<9} {:>6.2}  (random-layer baseline)",
            "mean",
            meanq / nt as f64
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §39 — first end-to-end BLIND readout: spread-weighted layers × all tokens.
    //
    //  §38 found raw_std (a layer's score spread over the 93 candidates) is a label-free
    //  proxy for that layer's discriminative power. §39 uses it for real: aggregate the
    //  per-def z over ALL the call's tokens, weighting each layer by its spread — NO
    //  label anywhere (no target z, no hand-picked layers). Variants:
    //    A sum_z     Σ_tok Σ_layer z              (all layers equal)
    //    B std·z     Σ_tok Σ_layer std·z          (spread-weighted)
    //    C std²·z    Σ_tok Σ_layer std²·z         (spread-weighted harder)
    //    D topK_std  Σ_tok Σ_{top-K std layers} z (hard-select high-spread layers)
    //  Report the target's honest rank for each (direction is a single global bit,
    //  calibratable on train; we show the target's natural direction). Run
    //  `S21_ONLY=1 S39=1` (`S39_CASE`, `S39_K` for D's layer count).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S39").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S39_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let kk: usize = std::env::var("S39_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        // Per token: the four variant contributions over defs. Summed across tokens.
        let zero4 = || {
            [
                vec![0f64; n_defs],
                vec![0f64; n_defs],
                vec![0f64; n_defs],
                vec![0f64; n_defs],
            ]
        };
        let totals = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                // per-layer z over defs + raw std.
                let mut zl: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
                let mut stds: Vec<f64> = Vec::with_capacity(n_layers);
                for l in 0..n_layers {
                    let scores: Vec<f32> = (0..n_defs)
                        .map(|di| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[di][l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, LAYER_P)]
                        })
                        .collect();
                    let (mean, std) = mean_std(&scores);
                    zl.push(
                        scores
                            .iter()
                            .map(|&s| (s - mean) as f64 / std.max(1e-6) as f64)
                            .collect(),
                    );
                    stds.push(std as f64);
                }
                // top-K layers by std (this token) for variant D.
                let mut by_std: Vec<usize> = (0..n_layers).collect();
                by_std.sort_by(|&a, &b| stds[b].partial_cmp(&stds[a]).unwrap());
                let topk: Vec<usize> = by_std[..kk.min(n_layers)].to_vec();
                let mut acc = zero4();
                for l in 0..n_layers {
                    for d in 0..n_defs {
                        let z = zl[l][d];
                        acc[0][d] += z;
                        acc[1][d] += stds[l] * z;
                        acc[2][d] += stds[l] * stds[l] * z;
                    }
                }
                for &l in &topk {
                    for d in 0..n_defs {
                        acc[3][d] += zl[l][d];
                    }
                }
                acc
            })
            .reduce(zero4, |mut a, b| {
                for v in 0..4 {
                    for d in 0..n_defs {
                        a[v][d] += b[v][d];
                    }
                }
                a
            });

        println!(
            "\n══ §39 — blind spread-weighted readout, all {} tokens (case {i}, tool={truth}) ══",
            resp.len()
        );
        println!(
            "  {:<10} {:>9} {:>4}  {:>7}  {:>9}",
            "variant", "tgt rank", "dir", "tgt z", "fam ⟨z⟩"
        );
        let names = ["A sum_z", "B std·z", "C std²·z", &format!("D top{kk}_std")];
        for v in 0..4 {
            let comb = &totals[v];
            let (mean, std) = {
                let m = comb.iter().sum::<f64>() / n_defs as f64;
                let sd = (comb.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n_defs as f64).sqrt();
                (m, sd.max(1e-9))
            };
            let mut tz = 0f64;
            let mut trank = n_defs;
            let mut tdir = "hi";
            for &td in &truth_defs {
                let z = (comb[td] - mean) / std;
                let rhi = 1 + comb.iter().filter(|&&v| v > comb[td]).count();
                let rlo = 1 + comb.iter().filter(|&&v| v < comb[td]).count();
                let (rank, dir) = if rhi <= rlo { (rhi, "hi") } else { (rlo, "lo") };
                if z.abs() > tz.abs() {
                    tz = z;
                    trank = rank;
                    tdir = dir;
                }
            }
            let fam_z = if family.is_empty() {
                0.0
            } else {
                family.iter().map(|&f| (comb[f] - mean) / std).sum::<f64>() / family.len() as f64
            };
            println!(
                "  {:<10} {:>5}/{:<3} {:>4}  {:>7.2}  {:>9.2}",
                names[v], trank, n_defs, tdir, tz, fam_z
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §40 — per-token HIT-COUNT vote (keep the §38 spread layer-selection).
    //
    //  §39 summed magnitudes across tokens. §40 votes instead: each decode token
    //  computes its blind score (top-K std layers by §38, z summed), picks its single
    //  best def (LOW direction — §39's calibration: target sits low), and that def gets
    //  a hit. Rank defs by hit count → the def the most tokens land on wins. Robust to
    //  noisy tokens (one vote each, not magnitude). Also a top-3 vote for the family.
    //  Run `S21_ONLY=1 S40=1` (`S40_CASE`, `S40_K` = std-layer count).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S40").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S40_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let kk: usize = std::env::var("S40_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();

        // Per token: its top-3 best defs (lowest blind score) → votes.
        let votes: Vec<[usize; 3]> = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                let mut zl: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
                let mut stds: Vec<f64> = Vec::with_capacity(n_layers);
                for l in 0..n_layers {
                    let scores: Vec<f32> = (0..n_defs)
                        .map(|di| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[di][l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, LAYER_P)]
                        })
                        .collect();
                    let (mean, std) = mean_std(&scores);
                    zl.push(
                        scores
                            .iter()
                            .map(|&s| (s - mean) as f64 / std.max(1e-6) as f64)
                            .collect(),
                    );
                    stds.push(std as f64);
                }
                let mut by_std: Vec<usize> = (0..n_layers).collect();
                by_std.sort_by(|&a, &b| stds[b].partial_cmp(&stds[a]).unwrap());
                let topk = &by_std[..kk.min(n_layers)];
                let tscore: Vec<f64> = (0..n_defs)
                    .map(|d| topk.iter().map(|&l| zl[l][d]).sum())
                    .collect();
                let mut order: Vec<usize> = (0..n_defs).collect();
                order.sort_by(|&a, &b| tscore[a].partial_cmp(&tscore[b]).unwrap()); // LOW = best
                [order[0], order[1], order[2]]
            })
            .collect();

        let nt = votes.len();
        let mut hits1 = vec![0u32; n_defs];
        let mut hits3 = vec![0u32; n_defs];
        for v in &votes {
            hits1[v[0]] += 1;
            for &h in v {
                hits3[h] += 1;
            }
        }
        let rank_of = |hits: &[u32], td: usize| -> usize {
            1 + hits.iter().filter(|&&h| h > hits[td]).count()
        };
        let mut order1: Vec<usize> = (0..n_defs).collect();
        order1.sort_by(|&a, &b| hits1[b].cmp(&hits1[a]));

        println!("\n══ §40 — per-token hit-count vote (case {i}, tool={truth}, {nt} tokens, top-{kk} std layers) ══");
        println!("  direction = lo · each token votes its single best def:");
        for (rank, &d) in order1.iter().take(8).enumerate() {
            if hits1[d] == 0 {
                break;
            }
            let mark = if truth_defs.contains(&d) {
                " ◀ TARGET"
            } else {
                ""
            };
            println!(
                "  {:>3}. {:<28} hits {:>3}/{}{}",
                rank + 1,
                def_sign[d].0,
                hits1[d],
                nt,
                mark
            );
        }
        for &td in &truth_defs {
            println!(
                "  target '{}':  top-1 vote rank {}/{} ({} hits)  ·  top-3 vote rank {}/{} ({} hits)",
                def_sign[td].0,
                rank_of(&hits1, td),
                n_defs,
                hits1[td],
                rank_of(&hits3, td),
                n_defs,
                hits3[td]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §41 — CONFIDENCE-weighted token aggregation.
    //
    //  §40's top-1 vote gave the target 0 (never the single best), so re-weighting
    //  votes can't help. Instead weight each token's SCORE CONTRIBUTION to the §39 sum
    //  by the token's confidence — keep "near-top accumulates" (got rank 6–7) but let
    //  confident/discriminative tokens dominate and generic ones fade. Confidence
    //  candidates: layer-spread (mean raw_std of the token's top-K std layers — §38 at
    //  the token level), def-spread (how separated the token's 93 def-scores are),
    //  margin (its top-1 vs top-2 gap). combined[def] = Σ_tok conf·(Σ_{topK} z[def]).
    //  Run `S21_ONLY=1 S41=1` (`S41_CASE`, `S41_K`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S41").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S41_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let kk: usize = std::env::var("S41_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        // Per token: (def scores from top-K std layers, layer-spread, def-spread, margin).
        let per_token: Vec<(Vec<f64>, f64, f64, f64)> = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                let mut zl: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
                let mut stds: Vec<f64> = Vec::with_capacity(n_layers);
                for l in 0..n_layers {
                    let scores: Vec<f32> = (0..n_defs)
                        .map(|di| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[di][l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, LAYER_P)]
                        })
                        .collect();
                    let (mean, std) = mean_std(&scores);
                    zl.push(
                        scores
                            .iter()
                            .map(|&s| (s - mean) as f64 / std.max(1e-6) as f64)
                            .collect(),
                    );
                    stds.push(std as f64);
                }
                let mut by_std: Vec<usize> = (0..n_layers).collect();
                by_std.sort_by(|&a, &b| stds[b].partial_cmp(&stds[a]).unwrap());
                let topk = &by_std[..kk.min(n_layers)];
                let layerspread = topk.iter().map(|&l| stds[l]).sum::<f64>() / kk as f64;
                let tscore: Vec<f64> = (0..n_defs)
                    .map(|d| topk.iter().map(|&l| zl[l][d]).sum())
                    .collect();
                let m = tscore.iter().sum::<f64>() / n_defs as f64;
                let defspread =
                    (tscore.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n_defs as f64).sqrt();
                let mut sorted = tscore.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let margin = sorted.get(1).copied().unwrap_or(0.0) - sorted[0]; // lo: 2nd − 1st ≥ 0
                (tscore, layerspread, defspread, margin)
            })
            .collect();

        type Wf = Box<dyn Fn(f64, f64, f64) -> f64>;
        let schemes: Vec<(&str, Wf)> = vec![
            ("uniform", Box::new(|_, _, _| 1.0)),
            ("layerspread", Box::new(|ls, _, _| ls)),
            ("defspread", Box::new(|_, ds, _| ds)),
            ("margin", Box::new(|_, _, m| m)),
            ("lspr*margin", Box::new(|ls, _, m| ls * m)),
            ("dspr*margin", Box::new(|_, ds, m| ds * m)),
        ];

        println!("\n══ §41 — confidence-weighted aggregation (case {i}, tool={truth}, {} tokens, top-{kk}) ══", resp.len());
        println!(
            "  {:<13} {:>9} {:>4}  {:>7}  {:>9}",
            "confidence", "tgt rank", "dir", "tgt z", "fam ⟨z⟩"
        );
        for (name, wf) in &schemes {
            let mut comb = vec![0f64; n_defs];
            for (ts, ls, ds, m) in &per_token {
                let w = wf(*ls, *ds, *m);
                for d in 0..n_defs {
                    comb[d] += w * ts[d];
                }
            }
            let mean = comb.iter().sum::<f64>() / n_defs as f64;
            let std = (comb.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_defs as f64)
                .sqrt()
                .max(1e-9);
            let mut tz = 0f64;
            let mut trank = n_defs;
            let mut tdir = "hi";
            for &td in &truth_defs {
                let z = (comb[td] - mean) / std;
                let rhi = 1 + comb.iter().filter(|&&v| v > comb[td]).count();
                let rlo = 1 + comb.iter().filter(|&&v| v < comb[td]).count();
                let (rank, dir) = if rhi <= rlo { (rhi, "hi") } else { (rlo, "lo") };
                if z.abs() > tz.abs() {
                    tz = z;
                    trank = rank;
                    tdir = dir;
                }
            }
            let fam_z = if family.is_empty() {
                0.0
            } else {
                family.iter().map(|&f| (comb[f] - mean) / std).sum::<f64>() / family.len() as f64
            };
            println!(
                "  {:<13} {:>5}/{:<3} {:>4}  {:>7.2}  {:>9.2}",
                name, trank, n_defs, tdir, tz, fam_z
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §42 — top-N hit count (each token adds +1 to each of its N best defs).
    //
    //  Per decode token: blind score (top-K std layers, §38), pick the N lowest-scoring
    //  defs (lo direction), +1 to each. After all tokens, rank defs by total hits. A
    //  softer vote than §40's top-1 — a def that is consistently *near* the top (top-N)
    //  every token accumulates, even if rarely #1. Full leaderboard shown (★=target,
    //  ·=family). Run `S21_ONLY=1 S42=1` (`S42_CASE`, `S42_K`=std layers, `S42_N`=top-N).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S42").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S42_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let kk: usize = std::env::var("S42_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let nn: usize = std::env::var("S42_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);

        let votes: Vec<Vec<usize>> = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                let mut zl: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
                let mut stds: Vec<f64> = Vec::with_capacity(n_layers);
                for l in 0..n_layers {
                    let scores: Vec<f32> = (0..n_defs)
                        .map(|di| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[di][l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, LAYER_P)]
                        })
                        .collect();
                    let (mean, std) = mean_std(&scores);
                    zl.push(
                        scores
                            .iter()
                            .map(|&s| (s - mean) as f64 / std.max(1e-6) as f64)
                            .collect(),
                    );
                    stds.push(std as f64);
                }
                let mut by_std: Vec<usize> = (0..n_layers).collect();
                by_std.sort_by(|&a, &b| stds[b].partial_cmp(&stds[a]).unwrap());
                let topk = &by_std[..kk.min(n_layers)];
                let tscore: Vec<f64> = (0..n_defs)
                    .map(|d| topk.iter().map(|&l| zl[l][d]).sum())
                    .collect();
                let mut order: Vec<usize> = (0..n_defs).collect();
                order.sort_by(|&a, &b| tscore[a].partial_cmp(&tscore[b]).unwrap()); // LOW = best
                order[..nn.min(n_defs)].to_vec()
            })
            .collect();

        let nt = votes.len();
        let mut hits = vec![0u32; n_defs];
        for v in &votes {
            for &d in v {
                hits[d] += 1;
            }
        }
        let mut order: Vec<usize> = (0..n_defs).collect();
        order.sort_by(|&a, &b| hits[b].cmp(&hits[a]));

        println!("\n══ §42 — top-{nn} hit count (case {i}, tool={truth}, {nt} tokens, top-{kk} std layers) ══");
        println!("  leaderboard (★=target, ·=family share stem '{tstem}'):");
        for (rank, &d) in order.iter().take(12).enumerate() {
            if hits[d] == 0 {
                break;
            }
            let mark = if truth_defs.contains(&d) {
                "★"
            } else if stem(&def_sign[d].0) == tstem {
                "·"
            } else {
                " "
            };
            println!(
                "  {} {:>3}. {:<28} hits {:>3}/{}",
                mark,
                rank + 1,
                def_sign[d].0,
                hits[d],
                nt
            );
        }
        for &td in &truth_defs {
            let rank = 1 + hits.iter().filter(|&&h| h > hits[td]).count();
            println!(
                "  target '{}': rank {}/{} ({} hits)",
                def_sign[td].0, rank, n_defs, hits[td]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §43 — WEIGHTED-sum scan: which layer weighting (by score-spread) wins?
    //
    //  Back to the §39 magnitude sum (best so far, rank 6–7), now scanning the per-layer
    //  weight as a function of that layer's score-spread (raw_std, the §38 selector):
    //  combined[def] = Σ_tok Σ_layer w(std_l)·z_l[def]. Tries uniform, std^{0.5,1,1.5,2},
    //  std-rank, top-{4,8,16} hard-select, std>median threshold, and relu of the
    //  standardized std. Reports each scheme's blind target rank, sorted best-first.
    //  Run `S21_ONLY=1 S43=1` (`S43_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S43").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S43_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        const NS: usize = 11;
        let snames = [
            "uniform",
            "sqrt(std)",
            "std",
            "std^1.5",
            "std^2",
            "std-rank",
            "top4",
            "top8",
            "top16",
            "std>med",
            "relu_zstd",
        ];
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        let zeros = || vec![vec![0f64; n_defs]; NS];
        let totals = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                let mut zl: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
                let mut stds: Vec<f64> = Vec::with_capacity(n_layers);
                for l in 0..n_layers {
                    let scores: Vec<f32> = (0..n_defs)
                        .map(|di| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[di][l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, LAYER_P)]
                        })
                        .collect();
                    let (mean, std) = mean_std(&scores);
                    zl.push(
                        scores
                            .iter()
                            .map(|&s| (s - mean) as f64 / std.max(1e-6) as f64)
                            .collect(),
                    );
                    stds.push(std as f64);
                }
                // derived: descending std rank, median, mean/sd of stds.
                let mut by_std: Vec<usize> = (0..n_layers).collect();
                by_std.sort_by(|&a, &b| stds[b].partial_cmp(&stds[a]).unwrap());
                let mut rank_desc = vec![0usize; n_layers];
                for (r, &l) in by_std.iter().enumerate() {
                    rank_desc[l] = r;
                }
                let mut ss = stds.clone();
                ss.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = ss[n_layers / 2];
                let smean = stds.iter().sum::<f64>() / n_layers as f64;
                let ssd = (stds.iter().map(|x| (x - smean).powi(2)).sum::<f64>() / n_layers as f64)
                    .sqrt()
                    .max(1e-9);
                let wof = |s: usize, l: usize| -> f64 {
                    let sd = stds[l];
                    match s {
                        0 => 1.0,
                        1 => sd.sqrt(),
                        2 => sd,
                        3 => sd.powf(1.5),
                        4 => sd * sd,
                        5 => (n_layers - rank_desc[l]) as f64 / n_layers as f64,
                        6 => (rank_desc[l] < 4) as u8 as f64,
                        7 => (rank_desc[l] < 8) as u8 as f64,
                        8 => (rank_desc[l] < 16) as u8 as f64,
                        9 => {
                            if sd > median {
                                sd
                            } else {
                                0.0
                            }
                        }
                        _ => ((sd - smean) / ssd).max(0.0),
                    }
                };
                let mut acc = zeros();
                for s in 0..NS {
                    for l in 0..n_layers {
                        let w = wof(s, l);
                        if w == 0.0 {
                            continue;
                        }
                        for d in 0..n_defs {
                            acc[s][d] += w * zl[l][d];
                        }
                    }
                }
                acc
            })
            .reduce(zeros, |mut a, b| {
                for s in 0..NS {
                    for d in 0..n_defs {
                        a[s][d] += b[s][d];
                    }
                }
                a
            });

        let mut rows: Vec<(usize, f64, &str, f64, &str)> = Vec::new(); // (rank, z, dir, famz, name)
        for s in 0..NS {
            let comb = &totals[s];
            let mean = comb.iter().sum::<f64>() / n_defs as f64;
            let std = (comb.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_defs as f64)
                .sqrt()
                .max(1e-9);
            let mut tz = 0f64;
            let mut trank = n_defs;
            let mut tdir = "hi";
            for &td in &truth_defs {
                let z = (comb[td] - mean) / std;
                let rhi = 1 + comb.iter().filter(|&&v| v > comb[td]).count();
                let rlo = 1 + comb.iter().filter(|&&v| v < comb[td]).count();
                let (rank, dir) = if rhi <= rlo { (rhi, "hi") } else { (rlo, "lo") };
                if z.abs() > tz.abs() {
                    tz = z;
                    trank = rank;
                    tdir = dir;
                }
            }
            let famz = if family.is_empty() {
                0.0
            } else {
                family.iter().map(|&f| (comb[f] - mean) / std).sum::<f64>() / family.len() as f64
            };
            rows.push((trank, tz, tdir, famz, snames[s]));
        }
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        println!(
            "\n══ §43 — layer-weighting scan, blind sum over {} tokens (case {i}, tool={truth}) ══",
            resp.len()
        );
        println!(
            "  {:<11} {:>9} {:>4} {:>7} {:>9}",
            "weighting", "tgt rank", "dir", "tgt z", "fam ⟨z⟩"
        );
        for (rank, z, dir, famz, name) in &rows {
            println!(
                "  {:<11} {:>5}/{:<3} {:>4} {:>7.2} {:>9.2}",
                name, rank, n_defs, dir, z, famz
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §44 — clamp the per-layer z: which sign of evidence to keep?
    //
    //  The layers split on direction (some +z, some −z); summing them partially cancels.
    //  Try clamping each per-def per-layer z before the blind sum over all tokens:
    //    raw  Σ z          (baseline)
    //    pos  Σ max(0,z)   (DITCH negative — keep only high-mismatch evidence)
    //    neg  Σ min(0,z)   (ditch positive — keep only good-match evidence)
    //    abs  Σ |z|        (magnitude of deviation, either way)
    //  Uniform layer weight to isolate the clamp. Run `S21_ONLY=1 S44=1` (`S44_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S44").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let i: usize = std::env::var("S44_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        const NS: usize = 4;
        let snames = ["raw Σz", "pos Σmax(0,z)", "neg Σmin(0,z)", "abs Σ|z|"];
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let tstem = stem(&truth);
        let family: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
            .map(|(j, _)| j)
            .collect();

        let zeros = || vec![vec![0f64; n_defs]; NS];
        let totals = resp
            .par_iter()
            .map(|&tk| {
                let q_sign = sign_pack(&tool_q_float[i][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                let mut acc = zeros();
                for l in 0..n_layers {
                    let scores: Vec<f32> = (0..n_defs)
                        .map(|di| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[di][l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, LAYER_P)]
                        })
                        .collect();
                    let (mean, std) = mean_std(&scores);
                    for d in 0..n_defs {
                        let z = (scores[d] - mean) as f64 / std.max(1e-6) as f64;
                        acc[0][d] += z;
                        acc[1][d] += z.max(0.0);
                        acc[2][d] += z.min(0.0);
                        acc[3][d] += z.abs();
                    }
                }
                acc
            })
            .reduce(zeros, |mut a, b| {
                for s in 0..NS {
                    for d in 0..n_defs {
                        a[s][d] += b[s][d];
                    }
                }
                a
            });

        println!(
            "\n══ §44 — z-clamp variants, blind sum over {} tokens (case {i}, tool={truth}) ══",
            resp.len()
        );
        println!(
            "  {:<14} {:>9} {:>4} {:>7} {:>9}",
            "variant", "tgt rank", "dir", "tgt z", "fam ⟨z⟩"
        );
        for s in 0..NS {
            let comb = &totals[s];
            let mean = comb.iter().sum::<f64>() / n_defs as f64;
            let std = (comb.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_defs as f64)
                .sqrt()
                .max(1e-9);
            let mut tz = 0f64;
            let mut trank = n_defs;
            let mut tdir = "hi";
            for &td in &truth_defs {
                let z = (comb[td] - mean) / std;
                let rhi = 1 + comb.iter().filter(|&&v| v > comb[td]).count();
                let rlo = 1 + comb.iter().filter(|&&v| v < comb[td]).count();
                let (rank, dir) = if rhi <= rlo { (rhi, "hi") } else { (rlo, "lo") };
                if z.abs() > tz.abs() {
                    tz = z;
                    trank = rank;
                    tdir = dir;
                }
            }
            let famz = if family.is_empty() {
                0.0
            } else {
                family.iter().map(|&f| (comb[f] - mean) / std).sum::<f64>() / family.len() as f64
            };
            println!(
                "  {:<14} {:>5}/{:<3} {:>4} {:>7.2} {:>9.2}",
                snames[s], trank, n_defs, tdir, tz, famz
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §45 — CROSS-CASE sweep of the blind readout (all 186 cases).
    //
    //  The §39–§44 readout, run on EVERY case: per token, p25-head / p75-layer mismatch
    //  → per-layer z, sqrt(std)-weighted (§38), summed over all the call's tokens; rank
    //  the target def (calibrated LOW direction). Reports Top-1/5/10 + median rank for
    //  the fixed-lo rule (and hi / per-case-best as references), plus a family-size
    //  breakdown — is rank driven by how many confusable siblings a tool has? This is
    //  the honest number telnet (rank 6, 8 siblings) was one hard point of. Run
    //  `S21_ONLY=1 S45=1` (`S45_MAXTOK` caps tokens/case, default all).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S45").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let maxtok: usize = std::env::var("S45_MAXTOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        // Per valid case: (rank_lo, rank_hi, family_size).
        let results: Vec<(usize, usize, usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                if truth_defs.is_empty() {
                    return None;
                }
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tstem = stem(truth);
                let famsize = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
                    .count();
                let mut comb = vec![0f64; n_defs];
                for &tk in resp.iter().take(maxtok) {
                    if tk >= tool_q_float[ci].len() {
                        continue;
                    }
                    let q_sign = sign_pack(&tool_q_float[ci][tk]);
                    let head_val: Vec<Vec<f32>> = def_sign
                        .iter()
                        .map(|(_, toks)| {
                            (0..n_heads)
                                .map(|hh| {
                                    let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                    let mut p: Vec<f32> = toks
                                        .iter()
                                        .map(|tw| {
                                            ((q_sign[wb] ^ tw[wb]).count_ones()
                                                + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                                as f32
                                        })
                                        .collect();
                                    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    p[pidx(p.len(), HEAD_P)]
                                })
                                .collect()
                        })
                        .collect();
                    for l in 0..n_layers {
                        let scores: Vec<f32> = (0..n_defs)
                            .map(|di| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[di][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect();
                        let (mean, std) = mean_std(&scores);
                        let w = (std as f64).sqrt();
                        for d in 0..n_defs {
                            comb[d] += w * (scores[d] - mean) as f64 / std.max(1e-6) as f64;
                        }
                    }
                }
                let mut rlo = n_defs;
                let mut rhi = n_defs;
                for &td in &truth_defs {
                    rlo = rlo.min(1 + comb.iter().filter(|&&v| v < comb[td]).count());
                    rhi = rhi.min(1 + comb.iter().filter(|&&v| v > comb[td]).count());
                }
                Some((rlo, rhi, famsize))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        let stats = |sel: &dyn Fn(&(usize, usize, usize)) -> usize| -> (f64, f64, f64, usize) {
            let mut ranks: Vec<usize> = results.iter().map(sel).collect();
            ranks.sort_unstable();
            let t1 = pct(ranks.iter().filter(|&&r| r == 1).count());
            let t5 = pct(ranks.iter().filter(|&&r| r <= 5).count());
            let t10 = pct(ranks.iter().filter(|&&r| r <= 10).count());
            let med = ranks[ranks.len() / 2];
            (t1, t5, t10, med)
        };
        println!(
            "\n══ §45 — cross-case blind readout: {} valid cases (of {n_cases}), {} defs ══",
            n, n_defs
        );
        println!("  metric: spread(sqrt-std)-weighted z-sum over all call tokens · chance T1/T5 = {:.1}/{:.1}%", pct(1), pct(5));
        println!(
            "  {:<14} {:>7} {:>7} {:>7} {:>8}",
            "direction", "Top-1%", "Top-5%", "Top-10%", "med rank"
        );
        for (name, f) in [
            (
                "fixed-lo",
                &(|r: &(usize, usize, usize)| r.0) as &dyn Fn(&(usize, usize, usize)) -> usize,
            ),
            ("fixed-hi", &(|r: &(usize, usize, usize)| r.1)),
            ("per-case best", &(|r: &(usize, usize, usize)| r.0.min(r.1))),
        ] {
            let (t1, t5, t10, med) = stats(f);
            println!(
                "  {:<14} {:>6.1} {:>7.1} {:>7.1} {:>8}",
                name, t1, t5, t10, med
            );
        }
        // Family-size breakdown (fixed-lo rank).
        println!("  fixed-lo rank by family size (siblings sharing the stem):");
        for (lo, hi, lbl) in [
            (0usize, 0usize, "0"),
            (1, 2, "1-2"),
            (3, 5, "3-5"),
            (6, usize::MAX, "6+"),
        ] {
            let sub: Vec<usize> = results
                .iter()
                .filter(|r| r.2 >= lo && r.2 <= hi)
                .map(|r| r.0)
                .collect();
            if sub.is_empty() {
                continue;
            }
            let mut s = sub.clone();
            s.sort_unstable();
            let t1 = 100.0 * s.iter().filter(|&&r| r == 1).count() as f64 / s.len() as f64;
            let t5 = 100.0 * s.iter().filter(|&&r| r <= 5).count() as f64 / s.len() as f64;
            println!(
                "    fam {:<4} n={:<4} Top-1 {:>5.1}%  Top-5 {:>5.1}%  med rank {}",
                lbl,
                s.len(),
                t1,
                t5,
                s[s.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §46 — per-head lock-on with exponential streak reward.
    //
    //  For every decode token, each head picks its single best def (lowest mismatch).
    //  Per (head, def) the n-th hit is worth 1,2,4,8,16,16,… (doubling, capped at 16) —
    //  so a head that KEEPS committing to the same def is rewarded exponentially. Total
    //  the escalated reward across all heads per def, rank for Top-1/Top-5. Cross-case.
    //  Run `S21_ONLY=1 S46=1` (`S46_MAXTOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S46").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let maxtok: usize = std::env::var("S46_MAXTOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        // n-th hit worth min(2^(n-1),16); cumulative reward for c hits by one head on one def.
        fn reward(c: u32) -> u64 {
            let mut total = 0u64;
            let mut inc = 1u64;
            for _ in 0..c {
                total += inc.min(16);
                inc = (inc * 2).min(16);
            }
            total
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        let results: Vec<(usize, usize, usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                if truth_defs.is_empty() {
                    return None;
                }
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tstem = stem(truth);
                let famsize = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
                    .count();
                let mut hits_lo = vec![0u32; n_heads * n_defs];
                let mut hits_hi = vec![0u32; n_heads * n_defs];
                for &tk in resp.iter().take(maxtok) {
                    if tk >= tool_q_float[ci].len() {
                        continue;
                    }
                    let q_sign = sign_pack(&tool_q_float[ci][tk]);
                    let head_val: Vec<Vec<f32>> = def_sign
                        .iter()
                        .map(|(_, toks)| {
                            (0..n_heads)
                                .map(|hh| {
                                    let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                    let mut p: Vec<f32> = toks
                                        .iter()
                                        .map(|tw| {
                                            ((q_sign[wb] ^ tw[wb]).count_ones()
                                                + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                                as f32
                                        })
                                        .collect();
                                    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    p[pidx(p.len(), HEAD_P)]
                                })
                                .collect()
                        })
                        .collect();
                    let _ = LAYER_P;
                    for hh in 0..n_heads {
                        let (mut blo, mut bhi) = (0usize, 0usize);
                        let (mut vlo, mut vhi) = (f32::MAX, f32::MIN);
                        for d in 0..n_defs {
                            let v = head_val[d][hh];
                            if v < vlo {
                                vlo = v;
                                blo = d;
                            }
                            if v > vhi {
                                vhi = v;
                                bhi = d;
                            }
                        }
                        hits_lo[hh * n_defs + blo] += 1;
                        hits_hi[hh * n_defs + bhi] += 1;
                    }
                }
                let mut tot_lo = vec![0u64; n_defs];
                let mut tot_hi = vec![0u64; n_defs];
                for hh in 0..n_heads {
                    for d in 0..n_defs {
                        tot_lo[d] += reward(hits_lo[hh * n_defs + d]);
                        tot_hi[d] += reward(hits_hi[hh * n_defs + d]);
                    }
                }
                let mut rlo = n_defs;
                let mut rhi = n_defs;
                for &td in &truth_defs {
                    rlo = rlo.min(1 + tot_lo.iter().filter(|&&v| v > tot_lo[td]).count());
                    rhi = rhi.min(1 + tot_hi.iter().filter(|&&v| v > tot_hi[td]).count());
                }
                Some((rlo, rhi, famsize))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        let stats = |sel: &dyn Fn(&(usize, usize, usize)) -> usize| -> (f64, f64, f64, usize) {
            let mut r: Vec<usize> = results.iter().map(sel).collect();
            r.sort_unstable();
            (
                pct(r.iter().filter(|&&x| x == 1).count()),
                pct(r.iter().filter(|&&x| x <= 5).count()),
                pct(r.iter().filter(|&&x| x <= 10).count()),
                r[r.len() / 2],
            )
        };
        println!("\n══ §46 — per-head lock-on streak reward: {} cases, {} defs (chance T1/T5 1.1/5.4%) ══", n, n_defs);
        println!(
            "  {:<14} {:>7} {:>7} {:>7} {:>8}",
            "direction", "Top-1%", "Top-5%", "Top-10%", "med rank"
        );
        for (name, f) in [
            (
                "lo (argmin)",
                &(|r: &(usize, usize, usize)| r.0) as &dyn Fn(&(usize, usize, usize)) -> usize,
            ),
            ("hi (argmax)", &(|r: &(usize, usize, usize)| r.1)),
            ("per-case best", &(|r: &(usize, usize, usize)| r.0.min(r.1))),
        ] {
            let (t1, t5, t10, med) = stats(f);
            println!(
                "  {:<14} {:>6.1} {:>7.1} {:>7.1} {:>8}",
                name, t1, t5, t10, med
            );
        }
        println!("  lo rank by family size:");
        for (lo, hi, lbl) in [
            (0usize, 0usize, "0"),
            (1, 2, "1-2"),
            (3, 5, "3-5"),
            (6, usize::MAX, "6+"),
        ] {
            let sub: Vec<usize> = results
                .iter()
                .filter(|r| r.2 >= lo && r.2 <= hi)
                .map(|r| r.0)
                .collect();
            if sub.is_empty() {
                continue;
            }
            let mut s = sub.clone();
            s.sort_unstable();
            let t1 = 100.0 * s.iter().filter(|&&r| r == 1).count() as f64 / s.len() as f64;
            let t5 = 100.0 * s.iter().filter(|&&r| r <= 5).count() as f64 / s.len() as f64;
            println!(
                "    fam {:<4} n={:<4} Top-1 {:>5.1}%  Top-5 {:>5.1}%  med rank {}",
                lbl,
                s.len(),
                t1,
                t5,
                s[s.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §47 — similar-K hit sharing (stop siblings stealing each other's hits).
    //
    //  Per token × per layer, get the top-1 hit def (lowest mismatch). Then boost EVERY
    //  def by its cosine similarity to the hit's K — so confusable defs (similar K, e.g.
    //  the session_list family) all get credited together instead of one stealing the
    //  vote. Def-def cosine from the mean def K (def_mw). Three variants: plain (no
    //  boost), soft (raw cosine), hard (cosine > thresh → +1). Cross-case Top-1/5.
    //  Run `S21_ONLY=1 S47=1` (`S47_THRESH` default 0.7).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S47").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let thresh: f64 = std::env::var("S47_THRESH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;

        // Def-def cosine similarity from the mean (non-structural) def K.
        let dnorm: Vec<Vec<f32>> = def_mw
            .iter()
            .map(|(_, v)| {
                let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter().map(|x| x / nrm).collect()
            })
            .collect();
        let smat: Vec<Vec<f64>> = (0..n_defs)
            .into_par_iter()
            .map(|a| {
                (0..n_defs)
                    .map(|b| {
                        dnorm[a]
                            .iter()
                            .zip(&dnorm[b])
                            .map(|(x, y)| x * y)
                            .sum::<f32>() as f64
                    })
                    .collect()
            })
            .collect();

        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let results: Vec<([usize; 3], usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                if truth_defs.is_empty() {
                    return None;
                }
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tstem = stem(truth);
                let famsize = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
                    .count();
                let mut sc = [vec![0f64; n_defs], vec![0f64; n_defs], vec![0f64; n_defs]];
                for &tk in resp.iter() {
                    if tk >= tool_q_float[ci].len() {
                        continue;
                    }
                    let q_sign = sign_pack(&tool_q_float[ci][tk]);
                    let head_val: Vec<Vec<f32>> = def_sign
                        .iter()
                        .map(|(_, toks)| {
                            (0..n_heads)
                                .map(|hh| {
                                    let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                    let mut p: Vec<f32> = toks
                                        .iter()
                                        .map(|tw| {
                                            ((q_sign[wb] ^ tw[wb]).count_ones()
                                                + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                                as f32
                                        })
                                        .collect();
                                    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    p[pidx(p.len(), HEAD_P)]
                                })
                                .collect()
                        })
                        .collect();
                    for l in 0..n_layers {
                        let ls: Vec<f32> = (0..n_defs)
                            .map(|d| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[d][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect();
                        let mut top1 = 0usize;
                        let mut best = f32::MAX;
                        for d in 0..n_defs {
                            if ls[d] < best {
                                best = ls[d];
                                top1 = d;
                            }
                        }
                        for d in 0..n_defs {
                            let s = smat[top1][d];
                            sc[0][d] += (d == top1) as u8 as f64;
                            sc[1][d] += s.max(0.0);
                            sc[2][d] += (s > thresh) as u8 as f64;
                        }
                    }
                }
                let rank = |score: &[f64]| -> usize {
                    let mut r = n_defs;
                    for &td in &truth_defs {
                        r = r.min(1 + score.iter().filter(|&&v| v > score[td]).count());
                    }
                    r
                };
                Some(([rank(&sc[0]), rank(&sc[1]), rank(&sc[2])], famsize))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!("\n══ §47 — similar-K hit sharing: {} cases, {} defs (chance T1/T5 1.1/5.4%), thresh={thresh} ══", n, n_defs);
        println!(
            "  {:<16} {:>7} {:>7} {:>7} {:>8}",
            "variant", "Top-1%", "Top-5%", "Top-10%", "med rank"
        );
        for (vi, name) in ["plain (no boost)", "soft (cosine)", "hard (>thresh)"]
            .iter()
            .enumerate()
        {
            let mut r: Vec<usize> = results.iter().map(|(ranks, _)| ranks[vi]).collect();
            r.sort_unstable();
            println!(
                "  {:<16} {:>6.1} {:>7.1} {:>7.1} {:>8}",
                name,
                pct(r.iter().filter(|&&x| x == 1).count()),
                pct(r.iter().filter(|&&x| x <= 5).count()),
                pct(r.iter().filter(|&&x| x <= 10).count()),
                r[r.len() / 2]
            );
        }
        println!("  soft-variant rank by family size:");
        for (lo, hi, lbl) in [
            (0usize, 0usize, "0"),
            (1, 2, "1-2"),
            (3, 5, "3-5"),
            (6, usize::MAX, "6+"),
        ] {
            let sub: Vec<usize> = results
                .iter()
                .filter(|(_, f)| *f >= lo && *f <= hi)
                .map(|(r, _)| r[1])
                .collect();
            if sub.is_empty() {
                continue;
            }
            let mut s = sub.clone();
            s.sort_unstable();
            let t1 = 100.0 * s.iter().filter(|&&r| r == 1).count() as f64 / s.len() as f64;
            let t5 = 100.0 * s.iter().filter(|&&r| r <= 5).count() as f64 / s.len() as f64;
            println!(
                "    fam {:<4} n={:<4} Top-1 {:>5.1}%  Top-5 {:>5.1}%  med rank {}",
                lbl,
                s.len(),
                t1,
                t5,
                s[s.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §48 — PER-LAYER similar-K hit sharing (§47 refined).
    //
    //  §47 boosted by GLOBAL mean-K cosine (Top-1 2.2). Here the boost uses the
    //  similarity at the HIT'S OWN LAYER — sharper, because the relevant K-neighborhood
    //  is tighter per layer. Per token × layer, get the top-1 hit, then boost every def
    //  by its per-layer K cosine to the hit. Variants: per-layer cosine, per-layer
    //  cosine², and the global cosine (§47 baseline). Cross-case Top-1/5.
    //  Run `S21_ONLY=1 S48=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S48").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;

        // Per-(def, layer) unit K slice from the mean (non-structural) def K.
        let unit: Vec<Vec<Vec<f32>>> = def_mw
            .iter()
            .map(|(_, v)| {
                (0..n_layers)
                    .map(|l| {
                        let sl = &v[l * PER_LAYER_DIM..(l + 1) * PER_LAYER_DIM];
                        let nrm = sl.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                        sl.iter().map(|x| x / nrm).collect()
                    })
                    .collect()
            })
            .collect();
        // Per-layer cosine matrix smat[l][a][b].
        let smat: Vec<Vec<Vec<f64>>> = (0..n_layers)
            .into_par_iter()
            .map(|l| {
                (0..n_defs)
                    .map(|a| {
                        (0..n_defs)
                            .map(|b| {
                                unit[a][l]
                                    .iter()
                                    .zip(&unit[b][l])
                                    .map(|(x, y)| x * y)
                                    .sum::<f32>() as f64
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        // Global mean-K cosine (the §47 baseline).
        let gnorm: Vec<Vec<f32>> = def_mw
            .iter()
            .map(|(_, v)| {
                let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter().map(|x| x / nrm).collect()
            })
            .collect();
        let gmat: Vec<Vec<f64>> = (0..n_defs)
            .into_par_iter()
            .map(|a| {
                (0..n_defs)
                    .map(|b| {
                        gnorm[a]
                            .iter()
                            .zip(&gnorm[b])
                            .map(|(x, y)| x * y)
                            .sum::<f32>() as f64
                    })
                    .collect()
            })
            .collect();

        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let results: Vec<([usize; 3], usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                if truth_defs.is_empty() {
                    return None;
                }
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tstem = stem(truth);
                let famsize = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
                    .count();
                let mut sc = [vec![0f64; n_defs], vec![0f64; n_defs], vec![0f64; n_defs]];
                for &tk in resp.iter() {
                    if tk >= tool_q_float[ci].len() {
                        continue;
                    }
                    let q_sign = sign_pack(&tool_q_float[ci][tk]);
                    let head_val: Vec<Vec<f32>> = def_sign
                        .iter()
                        .map(|(_, toks)| {
                            (0..n_heads)
                                .map(|hh| {
                                    let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                    let mut p: Vec<f32> = toks
                                        .iter()
                                        .map(|tw| {
                                            ((q_sign[wb] ^ tw[wb]).count_ones()
                                                + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                                as f32
                                        })
                                        .collect();
                                    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    p[pidx(p.len(), HEAD_P)]
                                })
                                .collect()
                        })
                        .collect();
                    for l in 0..n_layers {
                        let ls: Vec<f32> = (0..n_defs)
                            .map(|d| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[d][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect();
                        let mut top1 = 0usize;
                        let mut best = f32::MAX;
                        for d in 0..n_defs {
                            if ls[d] < best {
                                best = ls[d];
                                top1 = d;
                            }
                        }
                        for d in 0..n_defs {
                            let sl = smat[l][top1][d].max(0.0);
                            sc[0][d] += sl;
                            sc[1][d] += sl * sl;
                            sc[2][d] += gmat[top1][d].max(0.0);
                        }
                    }
                }
                let rank = |score: &[f64]| -> usize {
                    let mut r = n_defs;
                    for &td in &truth_defs {
                        r = r.min(1 + score.iter().filter(|&&v| v > score[td]).count());
                    }
                    r
                };
                Some(([rank(&sc[0]), rank(&sc[1]), rank(&sc[2])], famsize))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!("\n══ §48 — per-layer similar-K hit sharing: {} cases, {} defs (chance T1/T5 1.1/5.4%) ══", n, n_defs);
        println!(
            "  {:<18} {:>7} {:>7} {:>7} {:>8}",
            "variant", "Top-1%", "Top-5%", "Top-10%", "med rank"
        );
        for (vi, name) in [
            "per-layer cosine",
            "per-layer cosine²",
            "global cosine (§47)",
        ]
        .iter()
        .enumerate()
        {
            let mut r: Vec<usize> = results.iter().map(|(ranks, _)| ranks[vi]).collect();
            r.sort_unstable();
            println!(
                "  {:<18} {:>6.1} {:>7.1} {:>7.1} {:>8}",
                name,
                pct(r.iter().filter(|&&x| x == 1).count()),
                pct(r.iter().filter(|&&x| x <= 5).count()),
                pct(r.iter().filter(|&&x| x <= 10).count()),
                r[r.len() / 2]
            );
        }
        println!("  per-layer-cosine rank by family size:");
        for (lo, hi, lbl) in [
            (0usize, 0usize, "0"),
            (1, 2, "1-2"),
            (3, 5, "3-5"),
            (6, usize::MAX, "6+"),
        ] {
            let sub: Vec<usize> = results
                .iter()
                .filter(|(_, f)| *f >= lo && *f <= hi)
                .map(|(r, _)| r[0])
                .collect();
            if sub.is_empty() {
                continue;
            }
            let mut s = sub.clone();
            s.sort_unstable();
            let t1 = 100.0 * s.iter().filter(|&&r| r == 1).count() as f64 / s.len() as f64;
            let t5 = 100.0 * s.iter().filter(|&&r| r <= 5).count() as f64 / s.len() as f64;
            println!(
                "    fam {:<4} n={:<4} Top-1 {:>5.1}%  Top-5 {:>5.1}%  med rank {}",
                lbl,
                s.len(),
                t1,
                t5,
                s[s.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §49 — DECOUPLED similar-K boost: content-only cosine × all-token scoring.
    //
    //  §47/§48's def-def cosine came from the (all-token) mean K, which is scaffold-
    //  saturated (every pair ≈1). §49 computes the boost similarity from the CONTENT-only
    //  def mean K (def_content, structural always stripped) — a sharp neighborhood —
    //  while scoring still uses whatever token config is set (run with KEEP_STRUCT=1 for
    //  all-token scoring). Variants: content cosine (soft), the §47 all-token cosine
    //  (baseline), content cosine hard (>thresh, now meaningful). Cross-case Top-1/5.
    //  Run `S21_ONLY=1 KEEP_STRUCT=1 S49=1` (`S49_THRESH` default 0.5).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S49").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let thresh: f64 = std::env::var("S49_THRESH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;

        // cosine matrix from a set of def vectors.
        let cos_mat = |src: &[(String, Vec<f32>)]| -> Vec<Vec<f64>> {
            let unit: Vec<Vec<f32>> = src
                .iter()
                .map(|(_, v)| {
                    let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                    v.iter().map(|x| x / nrm).collect()
                })
                .collect();
            (0..n_defs)
                .into_par_iter()
                .map(|a| {
                    (0..n_defs)
                        .map(|b| {
                            unit[a]
                                .iter()
                                .zip(&unit[b])
                                .map(|(x, y)| x * y)
                                .sum::<f32>() as f64
                        })
                        .collect()
                })
                .collect()
        };
        let cmat = cos_mat(&def_content); // sharp, content-only
        let gmat = cos_mat(&def_mw); // §47 baseline (all-token if KEEP_STRUCT)

        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let results: Vec<([usize; 3], usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                if truth_defs.is_empty() {
                    return None;
                }
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tstem = stem(truth);
                let famsize = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(j, (t, _))| stem(t) == tstem && !truth_defs.contains(j))
                    .count();
                let mut sc = [vec![0f64; n_defs], vec![0f64; n_defs], vec![0f64; n_defs]];
                for &tk in resp.iter() {
                    if tk >= tool_q_float[ci].len() {
                        continue;
                    }
                    let q_sign = sign_pack(&tool_q_float[ci][tk]);
                    let head_val: Vec<Vec<f32>> = def_sign
                        .iter()
                        .map(|(_, toks)| {
                            (0..n_heads)
                                .map(|hh| {
                                    let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                    let mut p: Vec<f32> = toks
                                        .iter()
                                        .map(|tw| {
                                            ((q_sign[wb] ^ tw[wb]).count_ones()
                                                + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                                as f32
                                        })
                                        .collect();
                                    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    p[pidx(p.len(), HEAD_P)]
                                })
                                .collect()
                        })
                        .collect();
                    for l in 0..n_layers {
                        let ls: Vec<f32> = (0..n_defs)
                            .map(|d| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[d][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect();
                        let mut top1 = 0usize;
                        let mut best = f32::MAX;
                        for d in 0..n_defs {
                            if ls[d] < best {
                                best = ls[d];
                                top1 = d;
                            }
                        }
                        for d in 0..n_defs {
                            sc[0][d] += cmat[top1][d].max(0.0);
                            sc[1][d] += gmat[top1][d].max(0.0);
                            sc[2][d] += (cmat[top1][d] > thresh) as u8 as f64;
                        }
                    }
                }
                let rank = |score: &[f64]| -> usize {
                    let mut r = n_defs;
                    for &td in &truth_defs {
                        r = r.min(1 + score.iter().filter(|&&v| v > score[td]).count());
                    }
                    r
                };
                Some(([rank(&sc[0]), rank(&sc[1]), rank(&sc[2])], famsize))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!("\n══ §49 — decoupled content-cosine boost: {} cases, {} defs (chance T1/T5 1.1/5.4%), thresh={thresh} ══", n, n_defs);
        println!(
            "  {:<20} {:>7} {:>7} {:>7} {:>8}",
            "variant", "Top-1%", "Top-5%", "Top-10%", "med rank"
        );
        for (vi, name) in [
            "content cosine (soft)",
            "all-token cosine (§47)",
            "content hard (>thr)",
        ]
        .iter()
        .enumerate()
        {
            let mut r: Vec<usize> = results.iter().map(|(ranks, _)| ranks[vi]).collect();
            r.sort_unstable();
            println!(
                "  {:<20} {:>6.1} {:>7.1} {:>7.1} {:>8}",
                name,
                pct(r.iter().filter(|&&x| x == 1).count()),
                pct(r.iter().filter(|&&x| x <= 5).count()),
                pct(r.iter().filter(|&&x| x <= 10).count()),
                r[r.len() / 2]
            );
        }
        println!("  content-cosine rank by family size:");
        for (lo, hi, lbl) in [
            (0usize, 0usize, "0"),
            (1, 2, "1-2"),
            (3, 5, "3-5"),
            (6, usize::MAX, "6+"),
        ] {
            let sub: Vec<usize> = results
                .iter()
                .filter(|(_, f)| *f >= lo && *f <= hi)
                .map(|(r, _)| r[0])
                .collect();
            if sub.is_empty() {
                continue;
            }
            let mut s = sub.clone();
            s.sort_unstable();
            let t1 = 100.0 * s.iter().filter(|&&r| r == 1).count() as f64 / s.len() as f64;
            let t5 = 100.0 * s.iter().filter(|&&r| r <= 5).count() as f64 / s.len() as f64;
            println!(
                "    fam {:<4} n={:<4} Top-1 {:>5.1}%  Top-5 {:>5.1}%  med rank {}",
                lbl,
                s.len(),
                t1,
                t5,
                s[s.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §51 — §36 with the layer selection BAKED to score-spread (blind).
    //
    //  §36 ranked layers by |target z| and signed by sign(target z) — both use the
    //  label. §51 replaces the ranking with §38's blind score-spread (std), and tests
    //  blind signs too. Four variants, ceiling → fully blind:
    //    A |tz|-select · tz-sign          = §36 (label ceiling)
    //    B  std-select · tz-sign          (selection blind, sign still label)
    //    C  std-select · skew-sign        (BLIND — skew<0 ⇒ target a low outlier)
    //    D  std-select · global-low sign  (BLIND — assume target always low)
    //  Run `S21_ONLY=1 S51=1` (`S51_CASE`, `S51_TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S51").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let i: usize = std::env::var("S51_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S51_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();
        let head_val: Vec<Vec<f32>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p[pidx(p.len(), HEAD_P)]
                    })
                    .collect()
            })
            .collect();
        let layer_scores: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..n_defs)
                    .map(|d| {
                        let mut hs: Vec<f32> = (0..N_KV_HEAD)
                            .map(|h| head_val[d][l * N_KV_HEAD + h])
                            .collect();
                        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        hs[pidx(N_KV_HEAD, LAYER_P)]
                    })
                    .collect()
            })
            .collect();
        // per layer: (mean, std, target z, skew)
        let lstat: Vec<(f32, f32, f64, f64)> = (0..n_layers)
            .map(|l| {
                let (m, s) = mean_std(&layer_scores[l]);
                let zs: Vec<f64> = layer_scores[l]
                    .iter()
                    .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                    .collect();
                let mut tz = 0f64;
                for &td in &truth_defs {
                    if zs[td].abs() > tz.abs() {
                        tz = zs[td];
                    }
                }
                let skew = zs.iter().map(|z| z.powi(3)).sum::<f64>() / n_defs as f64;
                (m, s, tz, skew)
            })
            .collect();

        // rank_by_std: order key. sign_mode: 0=tz, 1=skew, 2=global-low.
        let run = |rank_by_std: bool, sign_mode: u8, label: &str| {
            let mut ord: Vec<usize> = (0..n_layers).collect();
            if rank_by_std {
                ord.sort_by(|&a, &b| lstat[b].1.partial_cmp(&lstat[a].1).unwrap());
            } else {
                ord.sort_by(|&a, &b| lstat[b].2.abs().partial_cmp(&lstat[a].2.abs()).unwrap());
            }
            let sgn = |l: usize| -> f64 {
                match sign_mode {
                    0 => {
                        if lstat[l].2 >= 0.0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    1 => {
                        if lstat[l].3 >= 0.0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ => -1.0,
                }
            };
            print!("  {:<42}", label);
            for &k in &[1usize, 2, 4, 8, 16, 24, 48] {
                let sel = &ord[..k.min(n_layers)];
                let comb: Vec<f64> = (0..n_defs)
                    .map(|d| {
                        sel.iter()
                            .map(|&l| {
                                let (m, s, _, _) = lstat[l];
                                sgn(l) * (layer_scores[l][d] - m) as f64 / s.max(1e-6) as f64
                            })
                            .sum()
                    })
                    .collect();
                let cm = comb.iter().sum::<f64>() / n_defs as f64;
                let mut crank = n_defs;
                let mut best = f64::MIN;
                for &td in &truth_defs {
                    let rank = 1 + comb.iter().filter(|&&v| v > comb[td]).count();
                    if comb[td] - cm > best {
                        best = comb[td] - cm;
                        crank = rank;
                    }
                }
                print!(" {:>5}", crank);
            }
            println!();
        };

        println!("\n══ §51 — §36 with blind (std) layer selection (case {i}, tool={truth}) ══");
        println!(
            "  target rank/93 at K = {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
            1, 2, 4, 8, 16, 24, 48
        );
        run(false, 0, "A |tz|-select · tz-sign (§36 ceiling)");
        run(true, 0, "B  std-select · tz-sign (semi-blind)");
        run(true, 1, "C  std-select · skew-sign (BLIND)");
        run(true, 2, "D  std-select · global-low (BLIND)");
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §52 — per-layer property battery, predicting target RANK (not |z|).
    //
    //  The math: std (variance) is biased toward family-CLUSTER layers (F outliers give
    //  F× the variance of a lone outlier), so std-selection picks the layers where the
    //  target is buried in its family. Kurtosis is the opposite — high for an ISOLATED
    //  outlier (≈N), low for a cluster (≈N/F). §52 gathers, per (case × layer), the
    //  target's true rank plus blind shape props, and reports which prop best separates
    //  the rank-1 layers (AUC). One token (mid) per case. Run `S21_ONLY=1 S52=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S52").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        const NP: usize = 6;
        let pnames = [
            "raw_std",
            "kurtosis",
            "max|z|",
            "gap1(top1-2)",
            "gap8(top8-9)",
            "Kiso8(div)",
        ];

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;

        // Def-def cosine (content-only K) for the K-isolation prop.
        let unit: Vec<Vec<f32>> = def_content
            .iter()
            .map(|(_, v)| {
                let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter().map(|x| x / nrm).collect()
            })
            .collect();
        let smat: Vec<Vec<f64>> = (0..n_defs)
            .into_par_iter()
            .map(|a| {
                (0..n_defs)
                    .map(|b| {
                        unit[a]
                            .iter()
                            .zip(&unit[b])
                            .map(|(x, y)| x * y)
                            .sum::<f32>() as f64
                    })
                    .collect()
            })
            .collect();

        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        // Per (case, layer): (target rank, [props]).
        let samples: Vec<(usize, [f64; NP])> = (0..n_cases)
            .into_par_iter()
            .flat_map_iter(|ci| {
                let mut out: Vec<(usize, [f64; NP])> = Vec::new();
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                let resp = &tool_ranges[ci][3];
                if truth_defs.is_empty() || resp.is_empty() {
                    return out.into_iter();
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return out.into_iter();
                }
                let q_sign = sign_pack(&tool_q_float[ci][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                for l in 0..n_layers {
                    let scores: Vec<f32> = (0..n_defs)
                        .map(|d| {
                            let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                .map(|h| head_val[d][l * N_KV_HEAD + h])
                                .collect();
                            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            hs[pidx(N_KV_HEAD, LAYER_P)]
                        })
                        .collect();
                    let (mean, std) = mean_std(&scores);
                    let zs: Vec<f64> = scores
                        .iter()
                        .map(|&s| (s - mean) as f64 / std.max(1e-6) as f64)
                        .collect();
                    let nf = n_defs as f64;
                    let kurt = zs.iter().map(|z| z.powi(4)).sum::<f64>() / nf - 3.0;
                    let mut absz: Vec<(f64, usize)> =
                        zs.iter().enumerate().map(|(d, &z)| (z.abs(), d)).collect();
                    absz.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let maxz = absz[0].0;
                    let gap1 = absz[0].0 - absz[1].0;
                    let gap8 = absz[7].0 - absz[8].0;
                    let top8: Vec<usize> = absz[..8].iter().map(|(_, d)| *d).collect();
                    let (mut sc, mut cnt) = (0.0, 0.0);
                    for a in 0..8 {
                        for b in a + 1..8 {
                            sc += smat[top8[a]][top8[b]];
                            cnt += 1.0;
                        }
                    }
                    let kiso = 1.0 - sc / cnt; // high = leaders diverse (isolated target)
                                               // target rank in its deviation direction (best over copies).
                    let mut rank = n_defs;
                    for &td in &truth_defs {
                        let tz = zs[td];
                        let r = if tz >= 0.0 {
                            1 + zs.iter().filter(|&&z| z > tz).count()
                        } else {
                            1 + zs.iter().filter(|&&z| z < tz).count()
                        };
                        rank = rank.min(r);
                    }
                    out.push((rank, [std as f64, kurt, maxz, gap1, gap8, kiso]));
                }
                out.into_iter()
            })
            .collect();

        // AUC (Mann-Whitney) that prop separates "rank ≤ thr" from the rest.
        let auc = |j: usize, thr: usize| -> f64 {
            let mut v: Vec<(f64, bool)> = samples.iter().map(|(r, p)| (p[j], *r <= thr)).collect();
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (n1, n0) = (
                v.iter().filter(|x| x.1).count(),
                v.iter().filter(|x| !x.1).count(),
            );
            if n1 == 0 || n0 == 0 {
                return 0.5;
            }
            let mut rsum = 0.0;
            let mut i = 0;
            while i < v.len() {
                let mut j2 = i;
                while j2 < v.len() && v[j2].0 == v[i].0 {
                    j2 += 1;
                }
                let avg_rank = (i + 1 + j2) as f64 / 2.0; // 1-based average rank over the tie block
                for k in i..j2 {
                    if v[k].1 {
                        rsum += avg_rank;
                    }
                }
                i = j2;
            }
            (rsum - n1 as f64 * (n1 as f64 + 1.0) / 2.0) / (n1 as f64 * n0 as f64)
        };

        let ns = samples.len();
        let n1 = samples.iter().filter(|(r, _)| *r == 1).count();
        let n5 = samples.iter().filter(|(r, _)| *r <= 5).count();
        println!("\n══ §52 — per-layer property → target-RANK predictiveness ══");
        println!("  {ns} (case×layer) samples · {n1} rank-1, {n5} rank≤5 layers · AUC>0.5 ⇒ prop predicts good layers");
        println!(
            "  {:<14} {:>10} {:>10}   {:>12} {:>12}",
            "property", "AUC(=1)", "AUC(≤5)", "mean@rank1", "mean@rest"
        );
        let mut rows: Vec<(f64, usize)> = (0..NP).map(|j| (auc(j, 1), j)).collect();
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (a1, j) in rows {
            let a5 = auc(j, 5);
            let (mut sg, mut ng, mut sr, mut nr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (r, p) in &samples {
                if *r == 1 {
                    sg += p[j];
                    ng += 1.0;
                } else {
                    sr += p[j];
                    nr += 1.0;
                }
            }
            println!(
                "  {:<14} {:>10.3} {:>10.3}   {:>12.3} {:>12.3}",
                pnames[j],
                a1,
                a5,
                sg / ng.max(1.0),
                sr / nr.max(1.0)
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §53 — cross-case: LOW-std vs HIGH-std layer selection (§52's flip).
    //
    //  §52 confirmed std is anti-predictive for rank (rank-1 layers have LOWER std), so
    //  flip §51's selection. Per case (one mid token), build the §36 signed combine
    //  under three orderings and report Top-1/5 across K:
    //    ceiling : |tz|-select · tz-sign   (label)
    //    hi-std  :  std↓-select · skew-sign (blind, §51)
    //    lo-std  :  std↑-select · skew-sign (blind, the §52 flip)
    //  Run `S21_ONLY=1 S53=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S53").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let ks = [1usize, 2, 4, 8];

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        // results[case] = [[rank; K]; 3 variants]
        let results: Vec<[[usize; 4]; 3]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                let resp = &tool_ranges[ci][3];
                if truth_defs.is_empty() || resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let q_sign = sign_pack(&tool_q_float[ci][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                // per layer: (std, tz, skew, zs[def])
                let layer: Vec<(f64, f64, f64, Vec<f64>)> = (0..n_layers)
                    .map(|l| {
                        let scores: Vec<f32> = (0..n_defs)
                            .map(|d| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[d][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect();
                        let (m, s) = mean_std(&scores);
                        let zs: Vec<f64> = scores
                            .iter()
                            .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                            .collect();
                        let mut tz = 0f64;
                        for &td in &truth_defs {
                            if zs[td].abs() > tz.abs() {
                                tz = zs[td];
                            }
                        }
                        let skew = zs.iter().map(|z| z.powi(3)).sum::<f64>() / n_defs as f64;
                        (s as f64, tz, skew, zs)
                    })
                    .collect();
                let rank_at = |order: &[usize], use_skew: bool, k: usize| -> usize {
                    let sel = &order[..k.min(n_layers)];
                    let comb: Vec<f64> = (0..n_defs)
                        .map(|d| {
                            sel.iter()
                                .map(|&l| {
                                    let s = if use_skew { layer[l].2 } else { layer[l].1 };
                                    let sg = if s >= 0.0 { 1.0 } else { -1.0 };
                                    sg * layer[l].3[d]
                                })
                                .sum()
                        })
                        .collect();
                    let mut r = n_defs;
                    for &td in &truth_defs {
                        r = r.min(1 + comb.iter().filter(|&&v| v > comb[td]).count());
                    }
                    r
                };
                let mut ord_tz: Vec<usize> = (0..n_layers).collect();
                ord_tz.sort_by(|&a, &b| layer[b].1.abs().partial_cmp(&layer[a].1.abs()).unwrap());
                let mut ord_hi: Vec<usize> = (0..n_layers).collect();
                ord_hi.sort_by(|&a, &b| layer[b].0.partial_cmp(&layer[a].0).unwrap());
                let mut ord_lo: Vec<usize> = (0..n_layers).collect();
                ord_lo.sort_by(|&a, &b| layer[a].0.partial_cmp(&layer[b].0).unwrap());
                let mut r = [[0usize; 4]; 3];
                for (ki, &k) in ks.iter().enumerate() {
                    r[0][ki] = rank_at(&ord_tz, false, k); // ceiling
                    r[1][ki] = rank_at(&ord_hi, true, k); // hi-std blind
                    r[2][ki] = rank_at(&ord_lo, true, k); // lo-std blind
                }
                Some(r)
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!(
            "\n══ §53 — low-std vs high-std layer selection, {} cases (chance T1/T5 1.1/5.4%) ══",
            n
        );
        let vnames = ["ceiling |tz|", "hi-std (blind)", "lo-std (blind)"];
        println!(
            "  {:<16}{}",
            "variant",
            ks.iter()
                .map(|k| format!("   K={k} T1/T5"))
                .collect::<String>()
        );
        for v in 0..3 {
            print!("  {:<16}", vnames[v]);
            for ki in 0..4 {
                let t1 = pct(results.iter().filter(|r| r[v][ki] == 1).count());
                let t5 = pct(results.iter().filter(|r| r[v][ki] <= 5).count());
                print!("  {:>4.1}/{:<4.1}", t1, t5);
            }
            println!();
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §54 — LEARNED blind layer-selector (close the §53 gap toward the ceiling).
    //
    //  §53: ceiling (label) Top-5 96%, blind lo-std 19.5%. The gap is selection quality.
    //  §54 fits a logistic model on TRAIN cases — P(layer ranks target ≤5) from blind
    //  shape features [−std, kurtosis, |skew|, max|z|, gap1, gap8, Kiso] + a learned
    //  per-layer prior — then on HELD-OUT cases picks the max-predicted layer and reads
    //  the target's (skew-signed) rank there. Baselines: lo-std, fixed-best-prior-layer,
    //  and the |tz| ceiling. 50/50 case split. Run `S21_ONLY=1 S54=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S54").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        const NF: usize = 7;

        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let unit: Vec<Vec<f32>> = def_content
            .iter()
            .map(|(_, v)| {
                let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter().map(|x| x / nrm).collect()
            })
            .collect();
        let smat: Vec<Vec<f64>> = (0..n_defs)
            .into_par_iter()
            .map(|a| {
                (0..n_defs)
                    .map(|b| {
                        unit[a]
                            .iter()
                            .zip(&unit[b])
                            .map(|(x, y)| x * y)
                            .sum::<f32>() as f64
                    })
                    .collect()
            })
            .collect();

        // Per layer: (features[NF], rank_blind (skew-sign), rank_tz (label-sign), std, |tz|).
        type Layer = ([f64; NF], usize, usize, f64, f64);
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let per_case: Vec<Option<Vec<Layer>>> = (0..n_cases)
            .into_par_iter()
            .map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                let resp = &tool_ranges[ci][3];
                if truth_defs.is_empty() || resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let q_sign = sign_pack(&tool_q_float[ci][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                let layers: Vec<Layer> = (0..n_layers)
                    .map(|l| {
                        let scores: Vec<f32> = (0..n_defs)
                            .map(|d| {
                                let mut hs: Vec<f32> = (0..N_KV_HEAD)
                                    .map(|h| head_val[d][l * N_KV_HEAD + h])
                                    .collect();
                                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                hs[pidx(N_KV_HEAD, LAYER_P)]
                            })
                            .collect();
                        let (m, s) = mean_std(&scores);
                        let zs: Vec<f64> = scores
                            .iter()
                            .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                            .collect();
                        let nf = n_defs as f64;
                        let kurt = zs.iter().map(|z| z.powi(4)).sum::<f64>() / nf - 3.0;
                        let skew = zs.iter().map(|z| z.powi(3)).sum::<f64>() / nf;
                        let mut absz: Vec<(f64, usize)> =
                            zs.iter().enumerate().map(|(d, &z)| (z.abs(), d)).collect();
                        absz.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        let (maxz, gap1, gap8) =
                            (absz[0].0, absz[0].0 - absz[1].0, absz[7].0 - absz[8].0);
                        let top8: Vec<usize> = absz[..8].iter().map(|(_, d)| *d).collect();
                        let (mut cs, mut cn) = (0.0, 0.0);
                        for a in 0..8 {
                            for b in a + 1..8 {
                                cs += smat[top8[a]][top8[b]];
                                cn += 1.0;
                            }
                        }
                        let kiso = 1.0 - cs / cn;
                        let (mut rb, mut rt, mut abstz) = (n_defs, n_defs, 0f64);
                        for &td in &truth_defs {
                            let z = zs[td];
                            // skew-signed (blind) rank
                            let r = if skew >= 0.0 {
                                1 + zs.iter().filter(|&&v| v > z).count()
                            } else {
                                1 + zs.iter().filter(|&&v| v < z).count()
                            };
                            rb = rb.min(r);
                            // label-signed (ceiling) rank
                            let rl = if z >= 0.0 {
                                1 + zs.iter().filter(|&&v| v > z).count()
                            } else {
                                1 + zs.iter().filter(|&&v| v < z).count()
                            };
                            rt = rt.min(rl);
                            if z.abs() > abstz {
                                abstz = z.abs();
                            }
                        }
                        (
                            [-(s as f64), kurt, skew.abs(), maxz, gap1, gap8, kiso],
                            rb,
                            rt,
                            s as f64,
                            abstz,
                        )
                    })
                    .collect();
                Some(layers)
            })
            .collect();
        let valid: Vec<usize> = (0..n_cases).filter(|&ci| per_case[ci].is_some()).collect();

        // 50/50 case split (even index = train).
        let train: Vec<usize> = valid.iter().copied().filter(|ci| ci % 2 == 0).collect();
        let test: Vec<usize> = valid.iter().copied().filter(|ci| ci % 2 == 1).collect();

        // Learned per-layer prior from train: P(rank_blind ≤ 5).
        let prior: Vec<f64> = (0..n_layers)
            .map(|l| {
                let c = train
                    .iter()
                    .filter(|&&ci| per_case[ci].as_ref().unwrap()[l].1 <= 5)
                    .count();
                c as f64 / train.len().max(1) as f64
            })
            .collect();

        // Standardise features over the train (case,layer) population (+ prior as feature NF).
        let mut mean = [0f64; NF + 1];
        let mut var = [0f64; NF + 1];
        let mut cnt = 0f64;
        for &ci in &train {
            for (l, ld) in per_case[ci].as_ref().unwrap().iter().enumerate() {
                for j in 0..NF {
                    mean[j] += ld.0[j];
                }
                mean[NF] += prior[l];
                cnt += 1.0;
            }
        }
        for m in mean.iter_mut() {
            *m /= cnt.max(1.0);
        }
        for &ci in &train {
            for (l, ld) in per_case[ci].as_ref().unwrap().iter().enumerate() {
                for j in 0..NF {
                    var[j] += (ld.0[j] - mean[j]).powi(2);
                }
                var[NF] += (prior[l] - mean[NF]).powi(2);
            }
        }
        let sd: Vec<f64> = var
            .iter()
            .map(|v| (v / cnt.max(1.0)).sqrt().max(1e-9))
            .collect();
        let feat = |ld: &Layer, l: usize| -> [f64; NF + 1] {
            let mut x = [0f64; NF + 1];
            for j in 0..NF {
                x[j] = (ld.0[j] - mean[j]) / sd[j];
            }
            x[NF] = (prior[l] - mean[NF]) / sd[NF];
            x
        };

        // Logistic regression (full-batch GD).
        let mut w = [0f64; NF + 2]; // weights + bias
        for _ in 0..400 {
            let mut grad = [0f64; NF + 2];
            let mut nn = 0f64;
            for &ci in &train {
                for (l, ld) in per_case[ci].as_ref().unwrap().iter().enumerate() {
                    let x = feat(ld, l);
                    let z = w[NF + 1] + (0..NF + 1).map(|j| x[j] * w[j]).sum::<f64>();
                    let p = 1.0 / (1.0 + (-z).exp());
                    let e = p - if ld.1 <= 5 { 1.0 } else { 0.0 };
                    for j in 0..NF + 1 {
                        grad[j] += e * x[j];
                    }
                    grad[NF + 1] += e;
                    nn += 1.0;
                }
            }
            for j in 0..NF + 2 {
                w[j] -= 0.5 * grad[j] / nn.max(1.0);
            }
        }

        // Evaluate on test: each selector picks one layer/case, read its rank.
        let best_prior_layer = (0..n_layers)
            .max_by(|&a, &b| prior[a].partial_cmp(&prior[b]).unwrap())
            .unwrap();
        let pick_eval = |sel: &dyn Fn(&[Layer]) -> usize, use_tz: bool| -> (f64, f64) {
            let (mut t1, mut t5) = (0.0, 0.0);
            for &ci in &test {
                let ld = per_case[ci].as_ref().unwrap();
                let l = sel(ld);
                let r = if use_tz { ld[l].2 } else { ld[l].1 };
                if r == 1 {
                    t1 += 1.0;
                }
                if r <= 5 {
                    t5 += 1.0;
                }
            }
            let n = test.len().max(1) as f64;
            (100.0 * t1 / n, 100.0 * t5 / n)
        };

        let (c1, c5) = pick_eval(
            &|ld| {
                (0..n_layers)
                    .max_by(|&a, &b| ld[a].4.partial_cmp(&ld[b].4).unwrap())
                    .unwrap()
            },
            true,
        );
        let (l1, l5) = pick_eval(
            &|ld| {
                (0..n_layers)
                    .min_by(|&a, &b| ld[a].3.partial_cmp(&ld[b].3).unwrap())
                    .unwrap()
            },
            false,
        );
        let (p1, p5) = pick_eval(&|_| best_prior_layer, false);
        let (m1, m5) = pick_eval(
            &|ld| {
                (0..n_layers)
                    .max_by(|&a, &b| {
                        let za =
                            w[NF + 1] + (0..NF + 1).map(|j| feat(&ld[a], a)[j] * w[j]).sum::<f64>();
                        let zb =
                            w[NF + 1] + (0..NF + 1).map(|j| feat(&ld[b], b)[j] * w[j]).sum::<f64>();
                        za.partial_cmp(&zb).unwrap()
                    })
                    .unwrap()
            },
            false,
        );

        println!(
            "\n══ §54 — learned blind layer-selector ({} train / {} test cases) ══",
            train.len(),
            test.len()
        );
        println!(
            "  best-prior layer = L{} (prior {:.2})  ·  chance T1/T5 1.1/5.4%",
            BAND_LO + best_prior_layer,
            prior[best_prior_layer]
        );
        println!("  {:<22} {:>8} {:>8}", "selector", "Top-1%", "Top-5%");
        println!("  {:<22} {:>8.1} {:>8.1}", "ceiling |tz| (label)", c1, c5);
        println!("  {:<22} {:>8.1} {:>8.1}", "lo-std (blind)", l1, l5);
        println!("  {:<22} {:>8.1} {:>8.1}", "fixed best-prior layer", p1, p5);
        println!("  {:<22} {:>8.1} {:>8.1}", "LEARNED LR (blind)", m1, m5);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §55 — FULL per-layer dump (find the blind marker of the winning layers).
    //
    //  One case, one token, every layer: the target's actual rank/z (the result) next
    //  to a wide battery of BLIND distribution properties — sorted by rank so the
    //  winning layers are at the top. Stare at it for what the rank-1 layers share.
    //  Run `S21_ONLY=1 S55=1` (`S55_CASE`, `S55_TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S55").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let i: usize = std::env::var("S55_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S55_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();

        let head_val: Vec<Vec<f32>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p[pidx(p.len(), HEAD_P)]
                    })
                    .collect()
            })
            .collect();

        // per layer: (rank, dir, tz, std, kurt, skew, maxz, gap1, gap2, gap8, nout, tgt_pctile, head_std_cv)
        let mut rows: Vec<(
            usize,
            char,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            usize,
            f64,
            f64,
            usize,
        )> = Vec::new();
        for l in 0..n_layers {
            // per-head value for this layer (the 4 heads), to measure head agreement.
            let hv: Vec<f32> = (0..N_KV_HEAD)
                .map(|h| {
                    let mut col: Vec<f32> = (0..n_defs)
                        .map(|d| head_val[d][l * N_KV_HEAD + h])
                        .collect();
                    col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    col[pidx(n_defs, 0.5)] // median over defs of this head — placeholder for scale
                })
                .collect();
            let _ = hv;
            let scores: Vec<f32> = (0..n_defs)
                .map(|d| {
                    let mut hs: Vec<f32> = (0..N_KV_HEAD)
                        .map(|h| head_val[d][l * N_KV_HEAD + h])
                        .collect();
                    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    hs[pidx(N_KV_HEAD, LAYER_P)]
                })
                .collect();
            let (m, s) = mean_std(&scores);
            let zs: Vec<f64> = scores
                .iter()
                .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                .collect();
            let nf = n_defs as f64;
            let kurt = zs.iter().map(|z| z.powi(4)).sum::<f64>() / nf - 3.0;
            let skew = zs.iter().map(|z| z.powi(3)).sum::<f64>() / nf;
            let mut sz: Vec<f64> = zs.clone();
            sz.sort_by(|a, b| a.partial_cmp(b).unwrap()); // ascending
            let mut az: Vec<f64> = zs.iter().map(|z| z.abs()).collect();
            az.sort_by(|a, b| b.partial_cmp(a).unwrap()); // |z| descending
            let (maxz, gap1, gap2, gap8) = (az[0], az[0] - az[1], az[1] - az[2], az[7] - az[8]);
            let nout = zs.iter().filter(|&&z| z.abs() > 2.0).count();
            // target: rank (best dir), z, and percentile position (0=lowest score).
            let (mut rank, mut tz, mut dir, mut pct) = (n_defs, 0f64, 'h', 0.0f64);
            for &td in &truth_defs {
                let z = zs[td];
                let (r, d) = if z >= 0.0 {
                    (1 + zs.iter().filter(|&&v| v > z).count(), 'h')
                } else {
                    (1 + zs.iter().filter(|&&v| v < z).count(), 'l')
                };
                if r < rank {
                    rank = r;
                    tz = z;
                    dir = d;
                    pct = zs.iter().filter(|&&v| v < z).count() as f64 / nf;
                }
            }
            // head agreement: coefficient of variation of the 4 heads' target values.
            let th: Vec<f64> = (0..N_KV_HEAD)
                .map(|h| head_val[truth_defs[0]][l * N_KV_HEAD + h] as f64)
                .collect();
            let hm = th.iter().sum::<f64>() / 4.0;
            let hsd = (th.iter().map(|x| (x - hm).powi(2)).sum::<f64>() / 4.0).sqrt();
            let hcv = hsd / hm.abs().max(1e-6);
            rows.push((
                rank,
                dir,
                tz,
                s as f64,
                kurt,
                skew,
                maxz,
                gap1,
                gap2,
                gap8,
                nout,
                pct,
                hcv,
                BAND_LO + l,
            ));
        }
        rows.sort_by_key(|r| r.0);
        println!(
            "\n══ §55 — full per-layer dump (case {i}, tool={truth}, token {tk}, {n_defs} defs) ══"
        );
        println!("  sorted by target rank. cols: L rank dir tz | std kurt skew maxz gap1 gap2 gap8 nout tpct hcv");
        for (rank, dir, tz, std, kurt, skew, maxz, gap1, gap2, gap8, nout, pct, hcv, lab) in &rows {
            println!(
                "  L{:<2} r{:>3} {} tz{:>+5.2} | std{:>5.2} kur{:>+5.1} sk{:>+5.2} mx{:>4.1} g1{:>4.2} g2{:>4.2} g8{:>4.2} no{:>2} tp{:>4.2} hcv{:>5.2}",
                lab, rank, dir, tz, std, kurt, skew, maxz, gap1, gap2, gap8, nout, pct, hcv
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §56 — FULL per-HEAD dump (192 heads = 48 layers × 4). The Water of Life.
    //
    //  Drop below the layer roll-up to every individual head: its 93-def distribution,
    //  the target's per-head rank/z, and blind shape props. Sorted by rank. Plus a
    //  summary: how many heads rank the target well, and which head-position (0–3) and
    //  layer-band the good heads cluster in. Run `S21_ONLY=1 S56=1` (`S56_CASE/TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S56").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        let i: usize = std::env::var("S56_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk: usize = std::env::var("S56_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);
        let truth_defs: Vec<usize> = def_sign
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| *t == truth)
            .map(|(j, _)| j)
            .collect();

        // head_val[def][head] = p25 over the def's tokens of the mismatch.
        let head_val: Vec<Vec<f32>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p[pidx(p.len(), HEAD_P)]
                    })
                    .collect()
            })
            .collect();

        // per head: (rank, dir, tz, std, kurt, skew, maxz, gap1, nout, tp, layer, head_in_layer)
        let mut rows: Vec<(
            usize,
            char,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            usize,
            f64,
            usize,
            usize,
        )> = Vec::new();
        for hh in 0..n_heads {
            let scores: Vec<f32> = (0..n_defs).map(|d| head_val[d][hh]).collect();
            let (m, s) = mean_std(&scores);
            let zs: Vec<f64> = scores
                .iter()
                .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                .collect();
            let nf = n_defs as f64;
            let kurt = zs.iter().map(|z| z.powi(4)).sum::<f64>() / nf - 3.0;
            let skew = zs.iter().map(|z| z.powi(3)).sum::<f64>() / nf;
            let mut az: Vec<f64> = zs.iter().map(|z| z.abs()).collect();
            az.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let (maxz, gap1) = (az[0], az[0] - az[1]);
            let nout = zs.iter().filter(|&&z| z.abs() > 2.0).count();
            let (mut rank, mut tz, mut dir, mut tp) = (n_defs, 0f64, 'h', 0.0f64);
            for &td in &truth_defs {
                let z = zs[td];
                let (r, d) = if z >= 0.0 {
                    (1 + zs.iter().filter(|&&v| v > z).count(), 'h')
                } else {
                    (1 + zs.iter().filter(|&&v| v < z).count(), 'l')
                };
                if r < rank {
                    rank = r;
                    tz = z;
                    dir = d;
                    tp = zs.iter().filter(|&&v| v < z).count() as f64 / nf;
                }
            }
            rows.push((
                rank,
                dir,
                tz,
                s as f64,
                kurt,
                skew,
                maxz,
                gap1,
                nout,
                tp,
                BAND_LO + hh / N_KV_HEAD,
                hh % N_KV_HEAD,
            ));
        }
        rows.sort_by_key(|r| r.0);
        println!(
            "\n══ §56 — full per-HEAD dump ({} heads · case {i}, tool={truth}, token {tk}) ══",
            n_heads
        );
        println!("  cols: L{{lyr}}h{{head}} rank dir tz | std kurt skew maxz gap1 nout tp");
        for (rank, dir, tz, std, kurt, skew, maxz, gap1, nout, tp, lyr, hd) in &rows {
            println!(
                "  L{:<2}h{} r{:>3} {} tz{:>+5.2} | std{:>5.2} kur{:>+5.1} sk{:>+5.2} mx{:>4.1} g1{:>4.2} no{:>2} tp{:>4.2}",
                lyr, hd, rank, dir, tz, std, kurt, skew, maxz, gap1, nout, tp
            );
        }
        // summary: good-head counts + head-position / band distribution.
        let good: Vec<&_> = rows.iter().filter(|r| r.0 <= 5).collect();
        let mut by_pos = [0usize; N_KV_HEAD];
        let mut by_band = [0usize; 6];
        for r in &good {
            by_pos[r.11] += 1;
            by_band[((r.10) * 6 / 48).min(5)] += 1;
        }
        println!("  ── {} heads rank target ≤5 (of {n_heads}) ──", good.len());
        println!("  by head-position 0..3: {:?}", by_pos);
        println!("  by layer-band (L0-7,8-15,..): {:?}", by_band);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §57 — per-def HEAD-APPEARANCE count (cash in the §56 abundance).
    //
    //  §56: the target is a skew-directed extreme at ~92/192 heads (4.4× the average
    //  def). So don't select a head — COUNT. Per head, take its skew-directed top-K
    //  defs; each gets +1. Rank defs by total head-votes across all 192 heads. The
    //  target's abundance should beat the field if it beats its siblings. Cross-case,
    //  K ∈ {1,3,5}. Run `S21_ONLY=1 S57=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S57").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let kset = [1usize, 3, 5];
        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        const HEAD_P: f32 = 0.25;
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        let results: Vec<[usize; 3]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let truth_defs: Vec<usize> = def_sign
                    .iter()
                    .enumerate()
                    .filter(|(_, (t, _))| t == truth)
                    .map(|(j, _)| j)
                    .collect();
                let resp = &tool_ranges[ci][3];
                if truth_defs.is_empty() || resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let q_sign = sign_pack(&tool_q_float[ci][tk]);
                let head_val: Vec<Vec<f32>> = def_sign
                    .iter()
                    .map(|(_, toks)| {
                        (0..n_heads)
                            .map(|hh| {
                                let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                                let mut p: Vec<f32> = toks
                                    .iter()
                                    .map(|tw| {
                                        ((q_sign[wb] ^ tw[wb]).count_ones()
                                            + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                            as f32
                                    })
                                    .collect();
                                p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                p[pidx(p.len(), HEAD_P)]
                            })
                            .collect()
                    })
                    .collect();
                // votes[K_idx][def]
                let mut votes = [vec![0u32; n_defs], vec![0u32; n_defs], vec![0u32; n_defs]];
                for hh in 0..n_heads {
                    let scores: Vec<f32> = (0..n_defs).map(|d| head_val[d][hh]).collect();
                    let (m, s) = mean_std(&scores);
                    let zs: Vec<f64> = scores
                        .iter()
                        .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                        .collect();
                    let skew = zs.iter().map(|z| z.powi(3)).sum::<f64>() / n_defs as f64;
                    // order defs in the skew direction (skew<0 ⇒ lowest first).
                    let mut ord: Vec<usize> = (0..n_defs).collect();
                    if skew >= 0.0 {
                        ord.sort_by(|&a, &b| zs[b].partial_cmp(&zs[a]).unwrap());
                    // high first
                    } else {
                        ord.sort_by(|&a, &b| zs[a].partial_cmp(&zs[b]).unwrap());
                        // low first
                    }
                    for (ki, &k) in kset.iter().enumerate() {
                        for &d in &ord[..k.min(n_defs)] {
                            votes[ki][d] += 1;
                        }
                    }
                }
                let mut r = [n_defs; 3];
                for ki in 0..3 {
                    for &td in &truth_defs {
                        r[ki] =
                            r[ki].min(1 + votes[ki].iter().filter(|&&v| v > votes[ki][td]).count());
                    }
                }
                Some(r)
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!(
            "\n══ §57 — per-def head-appearance count: {} cases (chance T1/T5 1.1/5.4%) ══",
            n
        );
        println!(
            "  {:<10} {:>8} {:>8} {:>9}",
            "vote top-K", "Top-1%", "Top-5%", "med rank"
        );
        for (ki, &k) in kset.iter().enumerate() {
            let mut rr: Vec<usize> = results.iter().map(|r| r[ki]).collect();
            rr.sort_unstable();
            println!(
                "  top-{:<6} {:>8.1} {:>8.1} {:>9}",
                k,
                pct(rr.iter().filter(|&&x| x == 1).count()),
                pct(rr.iter().filter(|&&x| x <= 5).count()),
                rr[rr.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §58 — DISTINCTIVE-DIM agreement (match the bit where defs differ from the crowd).
    //
    //  Every readout so far pooled tokens / compared whole defs, averaging away the one
    //  token that separates siblings. §58 instead works dim-by-dim on the def CONTENT
    //  sign: per dim, the consensus sign over all 93 defs; a def's DISTINCTIVE dims are
    //  where it bucks consensus (its identity). Score(def) = net agreement of the probe
    //  with the def AT ONLY its distinctive dims. The def whose identity the probe
    //  agrees with wins. Diagnostic dump for the target's family + cross-case Top-1/5.
    //  Run `S21_ONLY=1 S58=1` (`S58_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S58").is_ok() {
        use rayon::prelude::*;
        let n_defs = def_sign.len();
        let dim = def_content[0].1.len();
        // def content sign per dim.
        let dsign: Vec<Vec<bool>> = def_content
            .iter()
            .map(|(_, v)| v.iter().map(|&x| x >= 0.0).collect())
            .collect();
        // consensus sign per dim + how strong (fraction agreeing).
        let consensus: Vec<bool> = (0..dim)
            .map(|d| dsign.iter().filter(|s| s[d]).count() * 2 > n_defs)
            .collect();
        // per def: distinctive dims (where it bucks consensus) and its sign there.
        let distinctive: Vec<Vec<(usize, bool)>> = (0..n_defs)
            .map(|di| {
                (0..dim)
                    .filter(|&d| dsign[di][d] != consensus[d])
                    .map(|d| (d, dsign[di][d]))
                    .collect()
            })
            .collect();

        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        // score one probe (per-dim float Q) against all defs on their distinctive dims.
        let score_all = |qf: &[f32]| -> Vec<f64> {
            (0..n_defs)
                .map(|di| {
                    let dd = &distinctive[di];
                    if dd.is_empty() {
                        return 0.0;
                    }
                    let net: i64 = dd
                        .iter()
                        .map(|&(d, sg)| if (qf[d] >= 0.0) == sg { 1 } else { -1 })
                        .sum();
                    net as f64 / dd.len() as f64 // normalise: net agreement fraction over the def's distinctive dims
                })
                .collect()
        };

        // ── Diagnostic: the target's family roster on one case ─────────────────
        let i: usize = std::env::var("S58_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk = resp.get(resp.len() / 2).copied().unwrap_or(0);
        let truth = tool_phase_tool[i].clone();
        let tstem = stem(&truth);
        let scores = score_all(&tool_q_float[i][tk]);
        let mut order: Vec<usize> = (0..n_defs).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let target = def_sign.iter().position(|(t, _)| t == &truth).unwrap_or(0);
        let trank = 1 + scores.iter().filter(|&&v| v > scores[target]).count();
        println!("\n══ §58 — distinctive-dim agreement (case {i}, tool={truth}, {n_defs} defs) ══");
        println!(
            "  target '{truth}': score {:+.3}, #distinctive-dims {}, rank {}/{}",
            scores[target],
            distinctive[target].len(),
            trank,
            n_defs
        );
        println!("  top-10 defs by distinctive-agreement (★=target ·=family):");
        for (r, &d) in order.iter().take(10).enumerate() {
            let mk = if d == target {
                "★"
            } else if stem(&def_sign[d].0) == tstem {
                "·"
            } else {
                " "
            };
            println!(
                "  {} {:>2}. {:<28} score{:+.3}  ndist {}",
                mk,
                r + 1,
                def_sign[d].0,
                scores[d],
                distinctive[d].len()
            );
        }

        // ── Cross-case readout ─────────────────────────────────────────────────
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let ranks: Vec<usize> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let td = def_sign.iter().position(|(t, _)| t == truth)?;
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let sc = score_all(&tool_q_float[ci][tk]);
                Some(1 + sc.iter().filter(|&&v| v > sc[td]).count())
            })
            .collect();
        let n = ranks.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        let mut rr = ranks.clone();
        rr.sort_unstable();
        println!("  ── cross-case ({n} cases, chance T1/T5 1.1/5.4%) ──");
        println!(
            "  Top-1 {:.1}%  Top-5 {:.1}%  Top-10 {:.1}%  med rank {}",
            pct(rr.iter().filter(|&&x| x == 1).count()),
            pct(rr.iter().filter(|&&x| x <= 5).count()),
            pct(rr.iter().filter(|&&x| x <= 10).count()),
            rr[rr.len() / 2]
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §59 — RAW per-head popcount dump, target + family, every head.
    //
    //  No stats. The actual head_val (p25-over-def-tokens XOR-sign popcount) for the
    //  target and each `session_list` sibling, at all 192 heads, grouped by layer
    //  (best layer first) and by head within a layer (best head first). Read it all.
    //  Run `S21_ONLY=1 S59=1` (`S59_CASE`, `S59_TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S59").is_ok() {
        use rayon::prelude::*;
        const LW: usize = PER_LAYER_DIM / 64;
        const HW: usize = HEAD_DIM / 64;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        fn pidx(n: usize, p: f32) -> usize {
            ((((n.max(1) - 1) as f32) * p).round() as usize).min(n.saturating_sub(1))
        }
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        const HEAD_P: f32 = 0.25;
        const LAYER_P: f32 = 0.75;
        let i: usize = std::env::var("S59_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk = std::env::var("S59_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let q_sign = sign_pack(&tool_q_float[i][tk]);
        let tstem = stem(&truth);
        let target = def_sign.iter().position(|(t, _)| t == &truth).unwrap_or(0);
        // family (stem match), target first; short labels.
        let mut fam: Vec<usize> = vec![target];
        for (d, (t, _)) in def_sign.iter().enumerate() {
            if d != target && stem(t) == tstem {
                fam.push(d);
            }
        }
        let short =
            |n: &str| -> String { n.split('_').next().unwrap_or(n).chars().take(4).collect() };
        let fam_lbl: Vec<String> = fam.iter().map(|&d| short(&def_sign[d].0)).collect();

        let head_val: Vec<Vec<f32>> = def_sign
            .par_iter()
            .map(|(_, toks)| {
                (0..n_heads)
                    .map(|hh| {
                        let wb = (hh / N_KV_HEAD) * LW + (hh % N_KV_HEAD) * HW;
                        let mut p: Vec<f32> = toks
                            .iter()
                            .map(|tw| {
                                ((q_sign[wb] ^ tw[wb]).count_ones()
                                    + (q_sign[wb + 1] ^ tw[wb + 1]).count_ones())
                                    as f32
                            })
                            .collect();
                        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        p[pidx(p.len(), HEAD_P)]
                    })
                    .collect()
            })
            .collect();

        // layer scores (p75 over heads) + target layer-rank, to order layers.
        let layer_rank = |l: usize| -> usize {
            let ls: Vec<f32> = (0..n_defs)
                .map(|d| {
                    let mut hs: Vec<f32> = (0..N_KV_HEAD)
                        .map(|h| head_val[d][l * N_KV_HEAD + h])
                        .collect();
                    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    hs[pidx(N_KV_HEAD, LAYER_P)]
                })
                .collect();
            let (m, s) = mean_std(&ls);
            let z = (ls[target] - m) / s.max(1e-6);
            if z >= 0.0 {
                1 + ls.iter().filter(|&&v| v > ls[target]).count()
            } else {
                1 + ls.iter().filter(|&&v| v < ls[target]).count()
            }
        };
        let head_rank = |hh: usize| -> usize {
            let col: Vec<f32> = (0..n_defs).map(|d| head_val[d][hh]).collect();
            let (m, s) = mean_std(&col);
            let z = (col[target] - m) / s.max(1e-6);
            if z >= 0.0 {
                1 + col.iter().filter(|&&v| v > col[target]).count()
            } else {
                1 + col.iter().filter(|&&v| v < col[target]).count()
            }
        };
        let mut layers: Vec<(usize, usize)> = (0..n_layers).map(|l| (layer_rank(l), l)).collect();
        layers.sort_by_key(|x| x.0);

        println!("\n══ §59 — raw per-head popcounts: target + family (case {i}, tool={truth}, tok {tk}) ══");
        println!("  family (target first): {}", fam_lbl.join(" "));
        println!(
            "  per head: rank | {} | field min/med/max   (lower popcount = better match)",
            fam_lbl.join(" ")
        );
        for (lrank, l) in &layers {
            println!("─ L{} (layer rank {}) ─", BAND_LO + l, lrank);
            let mut heads: Vec<(usize, usize)> = (0..N_KV_HEAD)
                .map(|h| (head_rank(l * N_KV_HEAD + h), h))
                .collect();
            heads.sort_by_key(|x| x.0);
            for (hrank, h) in &heads {
                let hh = l * N_KV_HEAD + h;
                let vals: Vec<String> = fam
                    .iter()
                    .map(|&d| {
                        let v = head_val[d][hh] as i32;
                        if d == target {
                            format!("[{v:>3}]")
                        } else {
                            format!("{v:>3}")
                        }
                    })
                    .collect();
                let col: Vec<f32> = (0..n_defs).map(|d| head_val[d][hh]).collect();
                let mut sc = col.clone();
                sc.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "  h{} r{:>2} | {} | {:.0}/{:.0}/{:.0}",
                    h,
                    hrank,
                    vals.join(" "),
                    sc[0],
                    sc[n_defs / 2],
                    sc[n_defs - 1]
                );
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §60 — RAW per-head Q·K dump (real attention), target + family, every head.
    //
    //  Like §59 but the per-head value is the actual float attention score:
    //  Q_probe · K_def (per head, √128-scaled), K = the def's content-mean float K.
    //  HIGHER = stronger attention = better match. Grouped by layer (best first) and
    //  head within layer (best first). Read it all. Run `S21_ONLY=1 S60=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S60").is_ok() {
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        let i: usize = std::env::var("S60_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = &tool_ranges[i][3];
        let tk = std::env::var("S60_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get(resp.len() / 2).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let qf = &tool_q_float[i][tk];
        let tstem = stem(&truth);
        let target = def_sign.iter().position(|(t, _)| t == &truth).unwrap_or(0);
        let mut fam: Vec<usize> = vec![target];
        for (d, (t, _)) in def_sign.iter().enumerate() {
            if d != target && stem(t) == tstem {
                fam.push(d);
            }
        }
        let short =
            |n: &str| -> String { n.split('_').next().unwrap_or(n).chars().take(4).collect() };
        let fam_lbl: Vec<String> = fam.iter().map(|&d| short(&def_sign[d].0)).collect();
        let scale = (HEAD_DIM as f32).sqrt();

        // head_qk[def][hh] = Σ_{d in head} Q[d]·K_def[d] / √HEAD_DIM.
        let head_qk: Vec<Vec<f32>> = def_content
            .iter()
            .map(|(_, k)| {
                (0..n_heads)
                    .map(|hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM)
                            .map(|j| qf[base + j] * k[base + j])
                            .sum::<f32>()
                            / scale
                    })
                    .collect()
            })
            .collect();

        let layer_rank = |l: usize| -> usize {
            // layer Q·K = mean over its 4 heads (a simple layer attention readout).
            let ls: Vec<f32> = (0..n_defs)
                .map(|d| {
                    (0..N_KV_HEAD)
                        .map(|h| head_qk[d][l * N_KV_HEAD + h])
                        .sum::<f32>()
                        / N_KV_HEAD as f32
                })
                .collect();
            let (m, s) = mean_std(&ls);
            let z = (ls[target] - m) / s.max(1e-6);
            if z >= 0.0 {
                1 + ls.iter().filter(|&&v| v > ls[target]).count()
            } else {
                1 + ls.iter().filter(|&&v| v < ls[target]).count()
            }
        };
        let head_rank = |hh: usize| -> usize {
            let col: Vec<f32> = (0..n_defs).map(|d| head_qk[d][hh]).collect();
            let (m, s) = mean_std(&col);
            let z = (col[target] - m) / s.max(1e-6);
            if z >= 0.0 {
                1 + col.iter().filter(|&&v| v > col[target]).count()
            } else {
                1 + col.iter().filter(|&&v| v < col[target]).count()
            }
        };
        let mut layers: Vec<(usize, usize)> = (0..n_layers).map(|l| (layer_rank(l), l)).collect();
        layers.sort_by_key(|x| x.0);

        println!("\n══ §60 — raw per-head Q·K (real attention): target + family (case {i}, tool={truth}, tok {tk}) ══");
        println!("  family (target first): {}", fam_lbl.join(" "));
        println!(
            "  per head: rank | {} | field min/med/max   (HIGHER Q·K = better match)",
            fam_lbl.join(" ")
        );
        for (lrank, l) in &layers {
            println!("─ L{} (layer rank {}) ─", BAND_LO + l, lrank);
            let mut heads: Vec<(usize, usize)> = (0..N_KV_HEAD)
                .map(|h| (head_rank(l * N_KV_HEAD + h), h))
                .collect();
            heads.sort_by_key(|x| x.0);
            for (hrank, h) in &heads {
                let hh = l * N_KV_HEAD + h;
                let vals: Vec<String> = fam
                    .iter()
                    .map(|&d| {
                        let v = head_qk[d][hh];
                        if d == target {
                            format!("[{v:>+5.1}]")
                        } else {
                            format!("{v:>+5.1}")
                        }
                    })
                    .collect();
                let col: Vec<f32> = (0..n_defs).map(|d| head_qk[d][hh]).collect();
                let mut sc = col.clone();
                sc.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "  h{} r{:>2} | {} | {:+.1}/{:+.1}/{:+.1}",
                    h,
                    hrank,
                    vals.join(" "),
                    sc[0],
                    sc[n_defs / 2],
                    sc[n_defs - 1]
                );
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §61 — HEAD-CONSENSUS as a blind layer selector.
    //
    //  §56's clue: good layers have multiple heads independently agreeing on a winner;
    //  bad layers scatter. Blind, label-free. Per layer, each of its 4 heads names its
    //  top def (max Q·K); consensus = how many of the 4 agree on the modal def (0.25..1).
    //  Test: (a) does consensus predict the target's rank-1 layer (AUC)? (b) selecting
    //  the max-consensus layer, what Top-1/5? vs lo-spread and oracle. Cross-case.
    //  Run `S21_ONLY=1 S61=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S61").is_ok() {
        use rayon::prelude::*;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let scale = (HEAD_DIM as f32).sqrt();
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        // per case: Vec over layers of (consensus, target_rank_at_layer, std_of_layer)
        let per_case: Vec<Vec<(f64, usize, f64)>> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let target = def_sign.iter().position(|(t, _)| t == truth)?;
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let qf = &tool_q_float[ci][tk];
                // head_qk[def][hh]
                let head_qk: Vec<Vec<f32>> = def_content
                    .iter()
                    .map(|(_, k)| {
                        (0..n_heads)
                            .map(|hh| {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                (0..HEAD_DIM)
                                    .map(|j| qf[base + j] * k[base + j])
                                    .sum::<f32>()
                                    / scale
                            })
                            .collect()
                    })
                    .collect();
                let out: Vec<(f64, usize, f64)> = (0..n_layers)
                    .map(|l| {
                        // each head's top def (max Q·K), consensus = modal agreement / 4.
                        let tops: Vec<usize> = (0..N_KV_HEAD)
                            .map(|h| {
                                let hh = l * N_KV_HEAD + h;
                                (0..n_defs)
                                    .max_by(|&a, &b| {
                                        head_qk[a][hh].partial_cmp(&head_qk[b][hh]).unwrap()
                                    })
                                    .unwrap()
                            })
                            .collect();
                        let mut best_cnt = 0;
                        for &d in &tops {
                            let c = tops.iter().filter(|&&x| x == d).count();
                            best_cnt = best_cnt.max(c);
                        }
                        let consensus = best_cnt as f64 / N_KV_HEAD as f64;
                        // layer score per def = mean Q·K over heads; target rank (high=good).
                        let ls: Vec<f32> = (0..n_defs)
                            .map(|d| {
                                (0..N_KV_HEAD)
                                    .map(|h| head_qk[d][l * N_KV_HEAD + h])
                                    .sum::<f32>()
                                    / N_KV_HEAD as f32
                            })
                            .collect();
                        let rank = 1 + ls.iter().filter(|&&v| v > ls[target]).count();
                        let (_, s) = mean_std(&ls);
                        (consensus, rank, s as f64)
                    })
                    .collect();
                Some(out)
            })
            .collect();
        let per_case: Vec<&Vec<(f64, usize, f64)>> = per_case.iter().collect();

        // (a) AUC: does consensus rank the rank-1 layers above the rest?
        let mut pos: Vec<f64> = Vec::new();
        let mut neg: Vec<f64> = Vec::new();
        for lc in &per_case {
            for &(cons, rank, _) in lc.iter() {
                if rank == 1 {
                    pos.push(cons);
                } else {
                    neg.push(cons);
                }
            }
        }
        let auc = {
            let mut all: Vec<(f64, bool)> = pos
                .iter()
                .map(|&x| (x, true))
                .chain(neg.iter().map(|&x| (x, false)))
                .collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (n1, n0) = (pos.len(), neg.len());
            let mut rsum = 0.0;
            let mut i = 0;
            while i < all.len() {
                let mut j = i;
                while j < all.len() && all[j].0 == all[i].0 {
                    j += 1;
                }
                let ar = (i + 1 + j) as f64 / 2.0;
                for k in i..j {
                    if all[k].1 {
                        rsum += ar;
                    }
                }
                i = j;
            }
            if n1 == 0 || n0 == 0 {
                0.5
            } else {
                (rsum - n1 as f64 * (n1 as f64 + 1.0) / 2.0) / (n1 as f64 * n0 as f64)
            }
        };

        // (b) selectors: pick one layer/case, read target rank there.
        let n = per_case.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        let eval = |sel: &dyn Fn(&[(f64, usize, f64)]) -> usize| -> (f64, f64, usize) {
            let mut rr: Vec<usize> = per_case.iter().map(|lc| lc[sel(lc)].1).collect();
            rr.sort_unstable();
            (
                pct(rr.iter().filter(|&&x| x == 1).count()),
                pct(rr.iter().filter(|&&x| x <= 5).count()),
                rr[rr.len() / 2],
            )
        };
        let (cc1, cc5, ccm) = eval(&|lc| {
            (0..lc.len())
                .max_by(|&a, &b| lc[a].0.partial_cmp(&lc[b].0).unwrap())
                .unwrap()
        });
        let (ls1, ls5, lsm) = eval(&|lc| {
            (0..lc.len())
                .min_by(|&a, &b| lc[a].2.partial_cmp(&lc[b].2).unwrap())
                .unwrap()
        });
        let (or1, or5, orm) = eval(&|lc| (0..lc.len()).min_by_key(|&a| lc[a].1).unwrap());

        println!(
            "\n══ §61 — head-consensus blind layer selector ({n} cases, chance T1/T5 1.1/5.4%) ══"
        );
        println!(
            "  consensus → rank-1 layer  AUC = {:.3}  (0.5 = useless)",
            auc
        );
        println!(
            "  {:<22} {:>7} {:>7} {:>8}",
            "selector", "Top-1%", "Top-5%", "med rank"
        );
        println!(
            "  {:<22} {:>7.1} {:>7.1} {:>8}",
            "max-consensus (blind)", cc1, cc5, ccm
        );
        println!(
            "  {:<22} {:>7.1} {:>7.1} {:>8}",
            "lo-spread (blind)", ls1, ls5, lsm
        );
        println!(
            "  {:<22} {:>7.1} {:>7.1} {:>8}",
            "oracle best layer", or1, or5, orm
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §62 — CONSENSUS-WEIGHTED layer aggregation (exploit §61's AUC 0.973).
    //
    //  Head-consensus marks decisive layers (AUC 0.973). Don't pick one — aggregate
    //  all layers weighted by consensus, so noise layers (~0 weight) drop out and the
    //  decisive layers vote. Variants (cross-case, Q·K):
    //    A sum            Σ_l z_l                          (unweighted baseline)
    //    B cons·z         Σ_l consensus_l · z_l            (consensus-weighted)
    //    C filter≥t       Σ_{cons≥t} z_l                   (hard-filter decisive)
    //    D winner-vote    Σ_{cons≥t} consensus_l·[winner]  (decisive layers vote winner)
    //  Run `S21_ONLY=1 S62=1` (`S62_T` filter threshold, default 0.5).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S62").is_ok() {
        use rayon::prelude::*;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let scale = (HEAD_DIM as f32).sqrt();
        let thr: f64 = std::env::var("S62_T")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        let results: Vec<[usize; 4]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let target = def_sign.iter().position(|(t, _)| t == truth)?;
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let qf = &tool_q_float[ci][tk];
                let head_qk: Vec<Vec<f32>> = def_content
                    .iter()
                    .map(|(_, k)| {
                        (0..n_heads)
                            .map(|hh| {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                (0..HEAD_DIM)
                                    .map(|j| qf[base + j] * k[base + j])
                                    .sum::<f32>()
                                    / scale
                            })
                            .collect()
                    })
                    .collect();
                let mut comb = [
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                ];
                for l in 0..n_layers {
                    // layer per-def score = mean Q·K over heads; z-normalise.
                    let ls: Vec<f32> = (0..n_defs)
                        .map(|d| {
                            (0..N_KV_HEAD)
                                .map(|h| head_qk[d][l * N_KV_HEAD + h])
                                .sum::<f32>()
                                / N_KV_HEAD as f32
                        })
                        .collect();
                    let (m, s) = mean_std(&ls);
                    let z: Vec<f64> = ls
                        .iter()
                        .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                        .collect();
                    // head consensus + the layer winner (modal head argmax).
                    let tops: Vec<usize> = (0..N_KV_HEAD)
                        .map(|h| {
                            let hh = l * N_KV_HEAD + h;
                            (0..n_defs)
                                .max_by(|&a, &b| {
                                    head_qk[a][hh].partial_cmp(&head_qk[b][hh]).unwrap()
                                })
                                .unwrap()
                        })
                        .collect();
                    let (mut winner, mut wc) = (tops[0], 0usize);
                    for &d in &tops {
                        let c = tops.iter().filter(|&&x| x == d).count();
                        if c > wc {
                            wc = c;
                            winner = d;
                        }
                    }
                    let cons = wc as f64 / N_KV_HEAD as f64;
                    for d in 0..n_defs {
                        comb[0][d] += z[d];
                        comb[1][d] += cons * z[d];
                        if cons >= thr {
                            comb[2][d] += z[d];
                        }
                    }
                    if cons >= thr {
                        comb[3][winner] += cons;
                    }
                }
                let mut r = [0usize; 4];
                for v in 0..4 {
                    r[v] = 1 + comb[v].iter().filter(|&&x| x > comb[v][target]).count();
                }
                Some(r)
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!("\n══ §62 — consensus-weighted layer aggregation ({n} cases, chance T1/T5 1.1/5.4%, thr={thr}) ══");
        println!(
            "  {:<16} {:>7} {:>7} {:>8}",
            "variant", "Top-1%", "Top-5%", "med rank"
        );
        for (v, name) in ["A sum", "B cons·z", "C filter≥t", "D winner-vote"]
            .iter()
            .enumerate()
        {
            let mut rr: Vec<usize> = results.iter().map(|r| r[v]).collect();
            rr.sort_unstable();
            println!(
                "  {:<16} {:>7.1} {:>7.1} {:>8}",
                name,
                pct(rr.iter().filter(|&&x| x == 1).count()),
                pct(rr.iter().filter(|&&x| x <= 5).count()),
                rr[rr.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §63 — FAMILY-Top-1: blind-pick layers so the #1 def is in the right family.
    //
    //  Goal change: not telnet at rank 1, but ANY same-stem (`session_list`) def at
    //  rank 1 — i.e. retrieve the right neighborhood. Test blind layer-selection
    //  formulas and report Family-Top-1 (top def shares the target stem), Family-Top-3,
    //  and the family's best rank, broken down by family size. Run `S21_ONLY=1 S63=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S63").is_ok() {
        use rayon::prelude::*;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let scale = (HEAD_DIM as f32).sqrt();
        let thr: f64 = std::env::var("S63_T")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        let stems: Vec<String> = def_sign.iter().map(|(t, _)| stem(t)).collect();
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        // per case: (family_best_rank for each of 4 readouts, family_size)
        let results: Vec<([usize; 4], usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let _target = def_sign.iter().position(|(t, _)| t == truth)?;
                let ts = stem(truth);
                let famset: Vec<usize> = (0..n_defs).filter(|&d| stems[d] == ts).collect();
                let famsize = famset.len() - 1; // siblings (excl. self)
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let qf = &tool_q_float[ci][tk];
                let head_qk: Vec<Vec<f32>> = def_content
                    .iter()
                    .map(|(_, k)| {
                        (0..n_heads)
                            .map(|hh| {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                (0..HEAD_DIM)
                                    .map(|j| qf[base + j] * k[base + j])
                                    .sum::<f32>()
                                    / scale
                            })
                            .collect()
                    })
                    .collect();
                let mut comb = [
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                ];
                for l in 0..n_layers {
                    let ls: Vec<f32> = (0..n_defs)
                        .map(|d| {
                            (0..N_KV_HEAD)
                                .map(|h| head_qk[d][l * N_KV_HEAD + h])
                                .sum::<f32>()
                                / N_KV_HEAD as f32
                        })
                        .collect();
                    let (m, s) = mean_std(&ls);
                    let z: Vec<f64> = ls
                        .iter()
                        .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                        .collect();
                    let tops: Vec<usize> = (0..N_KV_HEAD)
                        .map(|h| {
                            let hh = l * N_KV_HEAD + h;
                            (0..n_defs)
                                .max_by(|&a, &b| {
                                    head_qk[a][hh].partial_cmp(&head_qk[b][hh]).unwrap()
                                })
                                .unwrap()
                        })
                        .collect();
                    let (mut winner, mut wc) = (tops[0], 0usize);
                    for &d in &tops {
                        let c = tops.iter().filter(|&&x| x == d).count();
                        if c > wc {
                            wc = c;
                            winner = d;
                        }
                    }
                    let cons = wc as f64 / N_KV_HEAD as f64;
                    for d in 0..n_defs {
                        comb[0][d] += z[d];
                        comb[1][d] += cons * z[d];
                        if cons >= thr {
                            comb[2][d] += z[d];
                        }
                    }
                    if cons >= thr {
                        comb[3][winner] += cons;
                    }
                }
                // family best rank for each readout = min over family members of their rank.
                let fam_rank = |sc: &[f64]| -> usize {
                    famset
                        .iter()
                        .map(|&fd| 1 + sc.iter().filter(|&&x| x > sc[fd]).count())
                        .min()
                        .unwrap_or(n_defs)
                };
                Some((
                    [
                        fam_rank(&comb[0]),
                        fam_rank(&comb[1]),
                        fam_rank(&comb[2]),
                        fam_rank(&comb[3]),
                    ],
                    famsize,
                ))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!("\n══ §63 — FAMILY-Top-1 by blind readout ({n} cases, thr={thr}) ══");
        println!("  metric: is the #1 def a same-stem (session_list etc.) tool?");
        println!(
            "  {:<16} {:>9} {:>9} {:>9}",
            "readout", "FamTop-1%", "FamTop-3%", "fam med"
        );
        for (v, name) in ["A sum", "B cons·z", "C filter≥t", "D winner-vote"]
            .iter()
            .enumerate()
        {
            let mut rr: Vec<usize> = results.iter().map(|r| r.0[v]).collect();
            rr.sort_unstable();
            println!(
                "  {:<16} {:>9.1} {:>9.1} {:>9}",
                name,
                pct(rr.iter().filter(|&&x| x == 1).count()),
                pct(rr.iter().filter(|&&x| x <= 3).count()),
                rr[rr.len() / 2]
            );
        }
        // breakdown by family size for the best (winner-vote).
        println!("  winner-vote FamTop-1 by family size:");
        for (lo, hi, lbl) in [
            (0usize, 0usize, "0"),
            (1, 2, "1-2"),
            (3, 5, "3-5"),
            (6, usize::MAX, "6+"),
        ] {
            let sub: Vec<usize> = results
                .iter()
                .filter(|(_, f)| *f >= lo && *f <= hi)
                .map(|(r, _)| r[3])
                .collect();
            if sub.is_empty() {
                continue;
            }
            let t1 = 100.0 * sub.iter().filter(|&&x| x == 1).count() as f64 / sub.len() as f64;
            println!(
                "    fam {:<4} n={:<4} FamTop-1 {:>5.1}%",
                lbl,
                sub.len(),
                t1
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §64 — Family-Top-1 + per-def NULL-Z (kill the promiscuous distractors).
    //
    //  §63: the family sits at rank ~8 but promiscuous defs steal #1. Fix: per-def
    //  null-z — each def's baseline = its mean score across all OTHER cases (leave-one-
    //  out, leak-free); subtract it so a def that scores high for everyone drops out.
    //  Base readouts: C (consensus-filter z-sum) and D (winner-vote). Norms: raw,
    //  sub-mean, z-score. Report Family-Top-1 + fam-size breakdown. Run `S21_ONLY=1 S64=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S64").is_ok() {
        use rayon::prelude::*;
        let n_layers = BAND_HI - BAND_LO;
        let n_heads = n_layers * N_KV_HEAD;
        let n_defs = def_sign.len();
        let scale = (HEAD_DIM as f32).sqrt();
        let thr: f64 = std::env::var("S64_T")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        let stems: Vec<String> = def_sign.iter().map(|(t, _)| stem(t)).collect();
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        // pass 1: per case → (famset, C-scores, D-scores)
        let cases: Vec<(Vec<usize>, usize, Vec<f64>, Vec<f64>)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                def_sign.iter().position(|(t, _)| t == truth)?;
                let ts = stem(truth);
                let famset: Vec<usize> = (0..n_defs).filter(|&d| stems[d] == ts).collect();
                let famsize = famset.len() - 1;
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let qf = &tool_q_float[ci][tk];
                let head_qk: Vec<Vec<f32>> = def_content
                    .iter()
                    .map(|(_, k)| {
                        (0..n_heads)
                            .map(|hh| {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                (0..HEAD_DIM)
                                    .map(|j| qf[base + j] * k[base + j])
                                    .sum::<f32>()
                                    / scale
                            })
                            .collect()
                    })
                    .collect();
                let mut c = vec![0f64; n_defs];
                let mut d = vec![0f64; n_defs];
                for l in 0..n_layers {
                    let ls: Vec<f32> = (0..n_defs)
                        .map(|x| {
                            (0..N_KV_HEAD)
                                .map(|h| head_qk[x][l * N_KV_HEAD + h])
                                .sum::<f32>()
                                / N_KV_HEAD as f32
                        })
                        .collect();
                    let (m, s) = mean_std(&ls);
                    let z: Vec<f64> = ls
                        .iter()
                        .map(|&v| (v - m) as f64 / s.max(1e-6) as f64)
                        .collect();
                    let tops: Vec<usize> = (0..N_KV_HEAD)
                        .map(|h| {
                            let hh = l * N_KV_HEAD + h;
                            (0..n_defs)
                                .max_by(|&a, &b| {
                                    head_qk[a][hh].partial_cmp(&head_qk[b][hh]).unwrap()
                                })
                                .unwrap()
                        })
                        .collect();
                    let (mut winner, mut wc) = (tops[0], 0usize);
                    for &x in &tops {
                        let cnt = tops.iter().filter(|&&y| y == x).count();
                        if cnt > wc {
                            wc = cnt;
                            winner = x;
                        }
                    }
                    let cons = wc as f64 / N_KV_HEAD as f64;
                    if cons >= thr {
                        for x in 0..n_defs {
                            c[x] += z[x];
                        }
                        d[winner] += cons;
                    }
                }
                Some((famset, famsize, c, d))
            })
            .collect();
        let nc = cases.len().max(1);

        // pass 2: per-def leave-one-out mean/std for each base readout (which: 0=C, 1=D).
        let loo_stats = |which: usize| -> (Vec<f64>, Vec<f64>) {
            let mut sum = vec![0f64; n_defs];
            let mut sq = vec![0f64; n_defs];
            for cse in &cases {
                let v = if which == 0 { &cse.2 } else { &cse.3 };
                for d in 0..n_defs {
                    sum[d] += v[d];
                    sq[d] += v[d] * v[d];
                }
            }
            (sum, sq)
        };
        let (c_sum, c_sq) = loo_stats(0);
        let (d_sum, d_sq) = loo_stats(1);

        // Family-Top-1 for a base readout under a normalisation.
        let famtop =
            |which: usize, sum: &[f64], sq: &[f64], norm: u8| -> (f64, Vec<(usize, usize)>) {
                let mut hits = 0.0;
                let mut detail: Vec<(usize, usize)> = Vec::new(); // (famsize, fam_rank)
                for cse in &cases {
                    let v = if which == 0 { &cse.2 } else { &cse.3 };
                    let sc: Vec<f64> = (0..n_defs)
                        .map(|d| {
                            let lm = (sum[d] - v[d]) / (nc as f64 - 1.0).max(1.0);
                            let lv = ((sq[d] - v[d] * v[d]) / (nc as f64 - 1.0).max(1.0) - lm * lm)
                                .max(1e-9);
                            match norm {
                                0 => v[d],
                                1 => v[d] - lm,
                                _ => (v[d] - lm) / lv.sqrt(),
                            }
                        })
                        .collect();
                    let fr = cse
                        .0
                        .iter()
                        .map(|&fd| 1 + sc.iter().filter(|&&x| x > sc[fd]).count())
                        .min()
                        .unwrap_or(n_defs);
                    if fr == 1 {
                        hits += 1.0;
                    }
                    detail.push((cse.1, fr));
                }
                (100.0 * hits / nc as f64, detail)
            };

        println!(
            "\n══ §64 — Family-Top-1 + per-def null-z ({nc} cases, chance ≈ 3.3%, thr={thr}) ══"
        );
        println!("  {:<22} {:>10}", "readout × norm", "FamTop-1%");
        let mut best_detail = Vec::new();
        for (name, which, sum, sq) in [
            ("C filter", 0usize, &c_sum, &c_sq),
            ("D winner", 1usize, &d_sum, &d_sq),
        ] {
            for (nm, norm) in [("raw", 0u8), ("sub-mean", 1), ("z-score", 2)] {
                let (ft, det) = famtop(which, sum, sq, norm);
                println!("  {:<22} {:>10.1}", format!("{name} · {nm}"), ft);
                if name == "D winner" && norm == 2 {
                    best_detail = det;
                }
            }
        }
        println!("  D winner · z-score, FamTop-1 by family size:");
        for (lo, hi, lbl) in [
            (0usize, 0usize, "0"),
            (1, 2, "1-2"),
            (3, 5, "3-5"),
            (6, usize::MAX, "6+"),
        ] {
            let sub: Vec<usize> = best_detail
                .iter()
                .filter(|(f, _)| *f >= lo && *f <= hi)
                .map(|(_, r)| *r)
                .collect();
            if sub.is_empty() {
                continue;
            }
            let t1 = 100.0 * sub.iter().filter(|&&x| x == 1).count() as f64 / sub.len() as f64;
            println!(
                "    fam {:<4} n={:<4} FamTop-1 {:>5.1}%",
                lbl,
                sub.len(),
                t1
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §65 — FULL attention simulation (real softmax over all def tokens).
    //
    //  Not a per-def dot product — the actual operation: the probe's Q attends over
    //  EVERY definition's every token at once, softmax(Q·K/√d) per head, and we sum the
    //  attention mass that lands on each definition. "Where does the call look?" Reports
    //  the attention distribution over defs + the correct def's rank, for the mid token
    //  and summed over all call tokens. Run `S21_ONLY=1 S65=1` (`S65_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S65").is_ok() {
        let band = BAND_HI - BAND_LO;
        let n_heads = band * N_KV_HEAD;
        let sq = (HEAD_DIM as f32).sqrt();
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        // Re-read each def's per-token float K (all tokens).
        eprintln!(
            "  §65: re-reading per-token def K for {} defs…",
            def_streams.len()
        );
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let i: usize = std::env::var("S65_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp = tool_ranges[i][3].clone();
        let truth = tool_phase_tool[i].clone();
        let tstem = stem(&truth);
        let target = def_ktok.iter().position(|(t, _)| t == &truth).unwrap_or(0);

        // Full attention for one probe Q: total attention mass per def, summed over heads.
        let attend = |qf: &[f32]| -> Vec<f64> {
            let mut total = vec![0f64; n_defs];
            for hh in 0..n_heads {
                let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                let mut logits: Vec<(usize, f32)> = Vec::new();
                let mut maxl = f32::MIN;
                for (di, (_, kt)) in def_ktok.iter().enumerate() {
                    for tok in kt {
                        let mut l = 0f32;
                        for j in 0..HEAD_DIM {
                            l += qf[base + j] * tok[base + j];
                        }
                        l /= sq;
                        logits.push((di, l));
                        if l > maxl {
                            maxl = l;
                        }
                    }
                }
                let exps: Vec<f64> = logits
                    .iter()
                    .map(|(_, l)| ((l - maxl) as f64).exp())
                    .collect();
                let s: f64 = exps.iter().sum::<f64>().max(1e-12);
                for ((di, _), e) in logits.iter().zip(&exps) {
                    total[*di] += e / s;
                }
            }
            total
        };

        let report = |label: &str, attn: &[f64]| {
            let mut ord: Vec<usize> = (0..n_defs).collect();
            ord.sort_by(|&a, &b| attn[b].partial_cmp(&attn[a]).unwrap());
            let trank = 1 + attn.iter().filter(|&&v| v > attn[target]).count();
            let tot: f64 = attn.iter().sum();
            println!(
                "  ── {label} ──  target '{truth}' attn {:.3} ({:.1}% of total), rank {}/{}",
                attn[target],
                100.0 * attn[target] / tot,
                trank,
                n_defs
            );
            println!("     top-10 by attention (★=target ·=family):");
            for (r, &d) in ord.iter().take(10).enumerate() {
                let mk = if d == target {
                    "★"
                } else if stem(&def_ktok[d].0) == tstem {
                    "·"
                } else {
                    " "
                };
                println!(
                    "     {} {:>2}. {:<28} attn {:.3} ({:.1}%)",
                    mk,
                    r + 1,
                    def_ktok[d].0,
                    attn[d],
                    100.0 * attn[d] / tot
                );
            }
        };

        println!("\n══ §65 — full attention simulation (case {i}, tool={truth}, {n_defs} defs, {n_heads} heads) ══");
        let tk = resp.get(resp.len() / 2).copied().unwrap_or(0);
        if tk < tool_q_float[i].len() {
            report(&format!("mid token {tk}"), &attend(&tool_q_float[i][tk]));
        }
        // summed over all call tokens
        let mut agg = vec![0f64; n_defs];
        let mut nt = 0;
        for &t in &resp {
            if t < tool_q_float[i].len() {
                let a = attend(&tool_q_float[i][t]);
                for d in 0..n_defs {
                    agg[d] += a[d];
                }
                nt += 1;
            }
        }
        report(&format!("summed over {nt} call tokens"), &agg);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §66 — per-DEF parallel attention, formula iteration toward 100% match.
    //
    //  §65: global softmax → loud defs win. Fix: attend to each def separately, and
    //  take out the K-norm (cosine), since raw Q·K is norm-dominated and mean-sub is a
    //  no-op for Q·K. Per (probe mid-token, def, head): cosine(Q,K_tok) over the def's
    //  tokens, pooled (max / mean / logsumexp), summed over heads. Cross-case Top-1 +
    //  Family-Top-1 for each formula. `S66_BAND=routing` restricts to L24–39.
    //  Run `S21_ONLY=1 S66=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S66").is_ok() {
        use rayon::prelude::*;
        let routing = std::env::var("S66_BAND")
            .map(|v| v == "routing")
            .unwrap_or(false);
        let (lo, hi) = if routing {
            (24 - BAND_LO, 40 - BAND_LO)
        } else {
            (0, BAND_HI - BAND_LO)
        };
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        eprintln!("  §66: re-reading per-token def K…");
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let stems: Vec<String> = def_ktok.iter().map(|(t, _)| stem(t)).collect();
        // Precompute per (def, token, head): cos-ready K-head (unit) — store unit K per head slice.
        // To bound memory we recompute the dot on the fly but precompute per-token-head norm.
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        const NF: usize = 4;
        let fnames = ["raw-max", "cos-max", "cos-mean", "cos-lse"];
        let results: Vec<([usize; NF], usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let target = def_ktok.iter().position(|(t, _)| t == truth)?;
                let ts = stem(truth);
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let qf = &tool_q_float[ci][tk];
                // per-head |Q|
                let qn: Vec<f32> = heads
                    .iter()
                    .map(|&hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM)
                            .map(|j| qf[base + j] * qf[base + j])
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-6)
                    })
                    .collect();
                let mut score = [
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                    vec![0f64; n_defs],
                ];
                for (di, (_, kt)) in def_ktok.iter().enumerate() {
                    for (hi2, &hh) in heads.iter().enumerate() {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        let (mut rawmax, mut cmax, mut csum, mut clse) =
                            (f32::MIN, f32::MIN, 0f64, 0f64);
                        for tok in kt {
                            let mut dot = 0f32;
                            let mut kn = 0f32;
                            for j in 0..HEAD_DIM {
                                let k = tok[base + j];
                                dot += qf[base + j] * k;
                                kn += k * k;
                            }
                            let cos = dot / (qn[hi2] * kn.sqrt().max(1e-6));
                            if dot > rawmax {
                                rawmax = dot;
                            }
                            if cos > cmax {
                                cmax = cos;
                            }
                            csum += cos as f64;
                            clse += (cos as f64).exp();
                        }
                        let nt = kt.len().max(1) as f64;
                        score[0][di] += rawmax as f64;
                        score[1][di] += cmax as f64;
                        score[2][di] += csum / nt;
                        score[3][di] += clse.ln();
                    }
                }
                let mut r = [0usize; NF];
                let mut famr = [0usize; NF];
                let _ = &mut famr;
                for f in 0..NF {
                    r[f] = 1 + score[f].iter().filter(|&&x| x > score[f][target]).count();
                }
                // family rank (min over same-stem) for the best formula reported separately;
                // store target rank + a family-rank flag in the second slot via famsize encode.
                let fam_best = (0..NF)
                    .map(|f| {
                        (0..n_defs)
                            .filter(|&d| stems[d] == ts)
                            .map(|fd| 1 + score[f].iter().filter(|&&x| x > score[f][fd]).count())
                            .min()
                            .unwrap_or(n_defs)
                    })
                    .collect::<Vec<_>>();
                // pack: return target ranks + (encode family-top1 for cos-max in famsize slot)
                Some((r, fam_best[1]))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!(
            "\n══ §66 — per-def parallel attention ({n} cases, band={}, chance T1 1.1%) ══",
            if routing { "L24-39" } else { "all-48" }
        );
        println!(
            "  {:<10} {:>8} {:>8} {:>9}",
            "formula", "Top-1%", "Top-5%", "med rank"
        );
        for f in 0..NF {
            let mut rr: Vec<usize> = results.iter().map(|r| r.0[f]).collect();
            rr.sort_unstable();
            println!(
                "  {:<10} {:>8.1} {:>8.1} {:>9}",
                fnames[f],
                pct(rr.iter().filter(|&&x| x == 1).count()),
                pct(rr.iter().filter(|&&x| x <= 5).count()),
                rr[rr.len() / 2]
            );
        }
        let fam1 = pct(results.iter().filter(|r| r.1 == 1).count());
        println!("  (cos-max Family-Top-1 = {:.1}%)", fam1);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §67 — per-def attention over ALL decode tokens (the whole tool-use).
    //
    //  §66 used one token; here the whole call attends. Per decode token: cosine-max
    //  attention to each def (sum over routing-band heads). Aggregate over tokens by
    //  SUM and by per-token VOTE (argmax def). Cross-case Top-1 + Family-Top-1.
    //  Run `S21_ONLY=1 S67=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S67").is_ok() {
        use rayon::prelude::*;
        let (lo, hi) = (24 - BAND_LO, 40 - BAND_LO);
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        eprintln!("  §67: re-reading per-token def K…");
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let stems: Vec<String> = def_ktok.iter().map(|(t, _)| stem(t)).collect();
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());

        let results: Vec<([usize; 2], [usize; 2], usize)> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let target = def_ktok.iter().position(|(t, _)| t == truth)?;
                let ts = stem(truth);
                let resp: Vec<usize> = tool_ranges[ci][3]
                    .iter()
                    .copied()
                    .filter(|&t| t < tool_q_float[ci].len())
                    .collect();
                if resp.is_empty() {
                    return None;
                }
                let mut sumsc = vec![0f64; n_defs];
                let mut votes = vec![0f64; n_defs];
                for &tk in &resp {
                    let qf = &tool_q_float[ci][tk];
                    let qn: Vec<f32> = heads
                        .iter()
                        .map(|&hh| {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            (0..HEAD_DIM)
                                .map(|j| qf[base + j] * qf[base + j])
                                .sum::<f32>()
                                .sqrt()
                                .max(1e-6)
                        })
                        .collect();
                    let mut tokdef = vec![0f64; n_defs];
                    for (di, (_, kt)) in def_ktok.iter().enumerate() {
                        let mut s = 0f64;
                        for (hi2, &hh) in heads.iter().enumerate() {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            let mut cmax = f32::MIN;
                            for tok in kt {
                                let (mut dot, mut kn) = (0f32, 0f32);
                                for j in 0..HEAD_DIM {
                                    let k = tok[base + j];
                                    dot += qf[base + j] * k;
                                    kn += k * k;
                                }
                                let cos = dot / (qn[hi2] * kn.sqrt().max(1e-6));
                                if cos > cmax {
                                    cmax = cos;
                                }
                            }
                            s += cmax as f64;
                        }
                        tokdef[di] = s;
                        sumsc[di] += s;
                    }
                    let am = (0..n_defs)
                        .max_by(|&a, &b| tokdef[a].partial_cmp(&tokdef[b]).unwrap())
                        .unwrap();
                    votes[am] += 1.0;
                }
                let rank = |sc: &[f64], d: usize| 1 + sc.iter().filter(|&&x| x > sc[d]).count();
                let famrank = |sc: &[f64]| {
                    (0..n_defs)
                        .filter(|&d| stems[d] == ts)
                        .map(|fd| rank(sc, fd))
                        .min()
                        .unwrap_or(n_defs)
                };
                Some((
                    [rank(&sumsc, target), rank(&votes, target)],
                    [famrank(&sumsc), famrank(&votes)],
                    0,
                ))
            })
            .collect();

        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!("\n══ §67 — attention over all decode tokens (routing band, {n} cases) ══");
        println!(
            "  {:<12} {:>8} {:>8} {:>11}",
            "aggregate", "Top-1%", "Top-5%", "FamTop-1%"
        );
        for (idx, name) in ["sum", "vote"].iter().enumerate() {
            let mut rr: Vec<usize> = results.iter().map(|r| r.0[idx]).collect();
            rr.sort_unstable();
            let ft = pct(results.iter().filter(|r| r.1[idx] == 1).count());
            println!(
                "  {:<12} {:>8.1} {:>8.1} {:>11.1}",
                name,
                pct(rr.iter().filter(|&&x| x == 1).count()),
                pct(rr.iter().filter(|&&x| x <= 5).count()),
                ft
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §68 — postfix [def | whole call], no causal mask, one attention pass.
    //
    //  All call tokens attend to def d's tokens (+ their constant self-term, which
    //  cancels across defs). So the def-attention mass that selects d is, per head,
    //  Σ_{call t} Σ_{def k} exp(Q_t·K_k/√d) — the soft attention onto the def. Rank defs
    //  by it (raw-exp = literal idea) and by Σ cos (norm-removed). Report telnet's
    //  ranking; if it works, expand. Run `S21_ONLY=1 S68=1` (`S68_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S68").is_ok() {
        use rayon::prelude::*;
        let (lo, hi) = (24 - BAND_LO, 40 - BAND_LO);
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        let sq = (HEAD_DIM as f32).sqrt();
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        eprintln!("  §68: re-reading per-token def K…");
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let i: usize = std::env::var("S68_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp: Vec<usize> = tool_ranges[i][3]
            .iter()
            .copied()
            .filter(|&t| t < tool_q_float[i].len())
            .collect();
        let truth = tool_phase_tool[i].clone();
        let tstem = stem(&truth);
        let target = def_ktok.iter().position(|(t, _)| t == &truth).unwrap_or(0);

        // per def: (sum exp, sum cos) over all call tokens × def tokens × heads.
        let scored: Vec<(f64, f64)> = (0..n_defs)
            .into_par_iter()
            .map(|di| {
                let kt = &def_ktok[di].1;
                let (mut sexp, mut scos) = (0f64, 0f64);
                for &tk in &resp {
                    let qf = &tool_q_float[i][tk];
                    for &hh in &heads {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        let qn = (0..HEAD_DIM)
                            .map(|j| qf[base + j] * qf[base + j])
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-6);
                        for tok in kt {
                            let (mut dot, mut kn) = (0f32, 0f32);
                            for j in 0..HEAD_DIM {
                                let k = tok[base + j];
                                dot += qf[base + j] * k;
                                kn += k * k;
                            }
                            let logit = (dot / sq).clamp(-30.0, 30.0);
                            sexp += (logit as f64).exp();
                            scos += (dot / (qn * kn.sqrt().max(1e-6))) as f64;
                        }
                    }
                }
                (sexp, scos)
            })
            .collect();

        let report = |label: &str, key: &dyn Fn(usize) -> f64| {
            let mut ord: Vec<usize> = (0..n_defs).collect();
            ord.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap());
            let trank = 1 + (0..n_defs).filter(|&d| key(d) > key(target)).count();
            println!("  ── {label} ──  target rank {}/{}", trank, n_defs);
            for (r, &d) in ord.iter().take(8).enumerate() {
                let mk = if d == target {
                    "★"
                } else if stem(&def_ktok[d].0) == tstem {
                    "·"
                } else {
                    " "
                };
                println!(
                    "     {} {:>2}. {:<28} {:.3}",
                    mk,
                    r + 1,
                    def_ktok[d].0,
                    key(d)
                );
            }
        };
        println!("\n══ §68 — postfix attention (case {i}, tool={truth}, {} call tokens, {n_defs} defs) ══", resp.len());
        report("raw-exp (Σ exp Q·K)", &|d| scored[d].0);
        report("cosine (Σ cos)", &|d| scored[d].1);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §69 — WHICH decode token carries the routing query?
    //
    //  Sweep every call token: cos-max attention to each def (routing band, summed over
    //  heads), report the target's rank at that token. The routing decision lives at a
    //  specific token (where the model commits to the name), not the mid-call. Find it.
    //  Run `S21_ONLY=1 S69=1` (`S69_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S69").is_ok() {
        use rayon::prelude::*;
        let (lo, hi) = (24 - BAND_LO, 40 - BAND_LO);
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        eprintln!("  §69: re-reading per-token def K…");
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let stems: Vec<String> = def_ktok.iter().map(|(t, _)| stem(t)).collect();
        let i: usize = std::env::var("S69_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp: Vec<usize> = tool_ranges[i][3]
            .iter()
            .copied()
            .filter(|&t| t < tool_q_float[i].len())
            .collect();
        let truth = tool_phase_tool[i].clone();
        let tstem = stem(&truth);
        let target = def_ktok.iter().position(|(t, _)| t == &truth).unwrap_or(0);

        // per resp token: (target rank, family best rank, top-1 def name)
        let rows: Vec<(usize, usize, usize, String)> = resp
            .par_iter()
            .map(|&tk| {
                let qf = &tool_q_float[i][tk];
                let qn: Vec<f32> = heads
                    .iter()
                    .map(|&hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM)
                            .map(|j| qf[base + j] * qf[base + j])
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-6)
                    })
                    .collect();
                let sc: Vec<f64> = (0..n_defs)
                    .map(|di| {
                        let kt = &def_ktok[di].1;
                        let mut s = 0f64;
                        for (hi2, &hh) in heads.iter().enumerate() {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            let mut cmax = f32::MIN;
                            for tok in kt {
                                let (mut dot, mut kn) = (0f32, 0f32);
                                for j in 0..HEAD_DIM {
                                    let k = tok[base + j];
                                    dot += qf[base + j] * k;
                                    kn += k * k;
                                }
                                let cos = dot / (qn[hi2] * kn.sqrt().max(1e-6));
                                if cos > cmax {
                                    cmax = cos;
                                }
                            }
                            s += cmax as f64;
                        }
                        s
                    })
                    .collect();
                let rank = 1 + sc.iter().filter(|&&x| x > sc[target]).count();
                let famr = (0..n_defs)
                    .filter(|&d| stems[d] == tstem)
                    .map(|fd| 1 + sc.iter().filter(|&&x| x > sc[fd]).count())
                    .min()
                    .unwrap_or(n_defs);
                let top1 = (0..n_defs)
                    .max_by(|&a, &b| sc[a].partial_cmp(&sc[b]).unwrap())
                    .unwrap();
                (tk, rank, famr, def_ktok[top1].0.clone())
            })
            .collect();

        println!(
            "\n══ §69 — routing query per decode token (case {i}, tool={truth}, {} tokens) ══",
            resp.len()
        );
        println!(
            "  {:>5} {:>8} {:>8}   {}",
            "tok", "tgtrank", "famrank", "top-1 def"
        );
        for (tk, rank, famr, top1) in &rows {
            let mk = if *rank == 1 {
                "★"
            } else if *famr == 1 {
                "·"
            } else {
                " "
            };
            println!("  {} {:>4} {:>8} {:>8}   {}", mk, tk, rank, famr, top1);
        }
        let best = rows.iter().min_by_key(|r| r.1).unwrap();
        println!(
            "  best token: tok {} → target rank {} (top-1 was '{}')",
            best.0, best.1, best.3
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §70 — dump the cos-max score distribution at a routing token.
    //
    //  §69 found tokens 78–93 rank telnet #1. Dump the full sorted cos-max scores at
    //  one such token to see if telnet is genuinely separated or in a degenerate tie,
    //  and what sits with it. Run `S21_ONLY=1 S70=1` (`S70_CASE`, `S70_TOK`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S70").is_ok() {
        use rayon::prelude::*;
        let (lo, hi) = (24 - BAND_LO, 40 - BAND_LO);
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        eprintln!("  §70: re-reading per-token def K…");
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let i: usize = std::env::var("S70_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let resp: Vec<usize> = tool_ranges[i][3]
            .iter()
            .copied()
            .filter(|&t| t < tool_q_float[i].len())
            .collect();
        let tk: usize = std::env::var("S70_TOK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| resp.get((resp.len() * 3) / 4).copied().unwrap_or(0));
        let truth = tool_phase_tool[i].clone();
        let tstem = stem(&truth);
        let target = def_ktok.iter().position(|(t, _)| t == &truth).unwrap_or(0);

        let qf = &tool_q_float[i][tk];
        let qn: Vec<f32> = heads
            .iter()
            .map(|&hh| {
                let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                (0..HEAD_DIM)
                    .map(|j| qf[base + j] * qf[base + j])
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6)
            })
            .collect();
        let sc: Vec<f64> = (0..n_defs)
            .into_par_iter()
            .map(|di| {
                let kt = &def_ktok[di].1;
                let mut s = 0f64;
                for (hi2, &hh) in heads.iter().enumerate() {
                    let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                    let mut cmax = f32::MIN;
                    for tok in kt {
                        let (mut dot, mut kn) = (0f32, 0f32);
                        for j in 0..HEAD_DIM {
                            let k = tok[base + j];
                            dot += qf[base + j] * k;
                            kn += k * k;
                        }
                        let cos = dot / (qn[hi2] * kn.sqrt().max(1e-6));
                        if cos > cmax {
                            cmax = cos;
                        }
                    }
                    s += cmax as f64;
                }
                s
            })
            .collect();
        let mut ord: Vec<usize> = (0..n_defs).collect();
        ord.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap());
        let trank = 1 + sc.iter().filter(|&&x| x > sc[target]).count();
        println!("\n══ §70 — cos-max score distribution at token {tk} (case {i}, tool={truth}) ══");
        println!(
            "  target rank {}/{}, score {:.4}.  top-20 (★=target ·=family):",
            trank, n_defs, sc[target]
        );
        for (r, &d) in ord.iter().take(20).enumerate() {
            let mk = if d == target {
                "★"
            } else if stem(&def_ktok[d].0) == tstem {
                "·"
            } else {
                " "
            };
            println!("  {} {:>2}. {:<28} {:.4}", mk, r + 1, def_ktok[d].0, sc[d]);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §71 — RoPE test: low-frequency dims should route, high-freq should be noise.
    //
    //  Cached Q/K are post-RoPE; the call→def relative position scrambles the HIGH-freq
    //  dim-pairs and spares the LOW-freq ones (small angle). candle `rope` splits a head
    //  into halves [0,64)/[64,128); pair i = (i, i+64), freq decreasing with i — so LOW
    //  freq = HIGH i. Cosine over dim subsets: full, low-freq (i≥48), high-freq (i<16).
    //  If low-freq ≫ full ≫ high-freq, RoPE is the culprit. Run `S21_ONLY=1 S71=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S71").is_ok() {
        use rayon::prelude::*;
        let (lo, hi) = (24 - BAND_LO, 40 - BAND_LO);
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        eprintln!("  §71: re-reading per-token def K…");
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        // dim subsets (within a head's 128): full, low-freq (i in 48..64), high-freq (i in 0..16).
        let subset_full: Vec<usize> = (0..HEAD_DIM).collect();
        let subset_lo: Vec<usize> = (48..64).chain(112..128).collect();
        let subset_hi: Vec<usize> = (0..16).chain(64..80).collect();
        let subsets = [
            ("full-128", &subset_full),
            ("low-freq", &subset_lo),
            ("high-freq", &subset_hi),
        ];

        let results: Vec<[usize; 3]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|ci| {
                let truth = tool_phase_tool[ci].as_str();
                let target = def_ktok.iter().position(|(t, _)| t == truth)?;
                let resp = &tool_ranges[ci][3];
                if resp.is_empty() {
                    return None;
                }
                let tk = resp[resp.len() / 2];
                if tk >= tool_q_float[ci].len() {
                    return None;
                }
                let qf = &tool_q_float[ci][tk];
                let mut r = [0usize; 3];
                for (si, (_, sub)) in subsets.iter().enumerate() {
                    let qn: Vec<f32> = heads
                        .iter()
                        .map(|&hh| {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            sub.iter()
                                .map(|&j| qf[base + j] * qf[base + j])
                                .sum::<f32>()
                                .sqrt()
                                .max(1e-6)
                        })
                        .collect();
                    let sc: Vec<f64> = (0..n_defs)
                        .map(|di| {
                            let kt = &def_ktok[di].1;
                            let mut s = 0f64;
                            for (hi2, &hh) in heads.iter().enumerate() {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                let mut cmax = f32::MIN;
                                for tok in kt {
                                    let (mut dot, mut kn) = (0f32, 0f32);
                                    for &j in sub.iter() {
                                        let k = tok[base + j];
                                        dot += qf[base + j] * k;
                                        kn += k * k;
                                    }
                                    let cos = dot / (qn[hi2] * kn.sqrt().max(1e-6));
                                    if cos > cmax {
                                        cmax = cos;
                                    }
                                }
                                s += cmax as f64;
                            }
                            s
                        })
                        .collect();
                    r[si] = 1 + sc.iter().filter(|&&x| x > sc[target]).count();
                }
                Some(r)
            })
            .collect();
        let n = results.len().max(1);
        let pct = |c: usize| 100.0 * c as f64 / n as f64;
        println!("\n══ §71 — RoPE dim-subset routing ({n} cases, chance T1 1.1%) ══");
        println!(
            "  {:<10} {:>8} {:>8} {:>9}",
            "dims", "Top-1%", "Top-5%", "med rank"
        );
        for (si, (name, _)) in subsets.iter().enumerate() {
            let mut rr: Vec<usize> = results.iter().map(|r| r[si]).collect();
            rr.sort_unstable();
            println!(
                "  {:<10} {:>8.1} {:>8.1} {:>9}",
                name,
                pct(rr.iter().filter(|&&x| x == 1).count()),
                pct(rr.iter().filter(|&&x| x <= 5).count()),
                rr[rr.len() / 2]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §72 — un-rotate RoPE, sweep the shared-prefix offset, recover content match.
    //
    //  Cached Q/K are post-RoPE. Invert it: def K token j → un-rotate by j; call Q token
    //  tk → un-rotate by (tk+QOFF). A common offset cancels in Q·K, so sweeping the single
    //  integer QOFF absorbs the (unknown) shared-prefix length. candle half-split rope:
    //  pair k=(k,k+64), freq f[k]=theta^(-2k/128). If RoPE is the wall, one QOFF makes the
    //  target snap to rank 1. One case. Run `S21_ONLY=1 S72=1` (`S72_THETA`,`S72_QMAX`,`S72_CASE`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S72").is_ok() {
        use rayon::prelude::*;
        let (lo, hi) = (24 - BAND_LO, 40 - BAND_LO);
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        let hd2 = HEAD_DIM / 2;
        let theta: f64 = std::env::var("S72_THETA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000_000.0);
        let qmax: usize = std::env::var("S72_QMAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let freq: Vec<f64> = (0..hd2)
            .map(|k| theta.powf(-2.0 * k as f64 / HEAD_DIM as f64))
            .collect();
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        // un-rotate a head's 128 dims at v[base..] by `pos`, into out[0..128] (content frame).
        let unrope = |v: &[f32], base: usize, pos: f64, freq: &[f64], out: &mut [f32]| {
            for k in 0..hd2 {
                let ang = pos * freq[k];
                let (cc, ss) = (ang.cos() as f32, ang.sin() as f32);
                let a = v[base + k];
                let b = v[base + k + hd2];
                out[k] = a * cc + b * ss;
                out[k + hd2] = b * cc - a * ss;
            }
        };
        eprintln!("  §72: re-reading per-token def K…");
        let mut def_ktok: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for (sid, name) in &def_streams {
            let Some(tb) = persistence.read_tokens(&substrate, *sid).ok().flatten() else {
                continue;
            };
            let Ok(toks) = decode_token_ids(&tb) else {
                continue;
            };
            let Some(kb) = read_token_bands(&mut persistence, &substrate, *sid, toks.len()) else {
                continue;
            };
            def_ktok.push((name.clone(), kb));
        }
        let n_defs = def_ktok.len();
        let i: usize = std::env::var("S72_CASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let i = i.min(tool_q_float.len().saturating_sub(1));
        let truth = tool_phase_tool[i].clone();
        let tstem = stem(&truth);
        let target = def_ktok.iter().position(|(t, _)| t == &truth).unwrap_or(0);
        let dimn = tool_q_float[i][0].len();

        // Precompute content (un-rotated) def K per (def, token) — independent of QOFF.
        // S72_NOKROT: leave K untouched (pos 0 = identity) — for the "K is already content,
        // un-rope ONLY Q" hypothesis. Default un-rotates K by its token index j.
        let krot = std::env::var("S72_NOKROT").is_err();
        eprintln!("  §72: preparing def K (krot={krot})…");
        let def_content: Vec<Vec<Vec<f32>>> = def_ktok
            .par_iter()
            .map(|(_, kt)| {
                kt.iter()
                    .enumerate()
                    .map(|(j, tok)| {
                        let mut buf = vec![0f32; dimn];
                        let mut hb = vec![0f32; HEAD_DIM];
                        let kpos = if krot { j as f64 } else { 0.0 };
                        for &hh in &heads {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            unrope(tok, base, kpos, &freq, &mut hb);
                            buf[base..base + HEAD_DIM].copy_from_slice(&hb);
                        }
                        buf
                    })
                    .collect()
            })
            .collect();
        // content call tokens (non-zero Q).
        let calltoks: Vec<usize> = tool_ranges[i][3]
            .iter()
            .copied()
            .filter(|&tk| {
                tk < tool_q_float[i].len() && {
                    let qf = &tool_q_float[i][tk];
                    heads.iter().any(|&hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM).map(|j| qf[base + j].abs()).sum::<f32>() > 1e-3
                    })
                }
            })
            .collect();
        eprintln!(
            "  §72: {} content call tokens, sweeping QOFF 0..{qmax}…",
            calltoks.len()
        );

        let sweep: Vec<(usize, usize, usize)> = (0..qmax)
            .into_par_iter()
            .map(|qoff| {
                let mut score = vec![0f64; n_defs];
                let mut qb = vec![0f32; HEAD_DIM];
                for &tk in &calltoks {
                    let qf = &tool_q_float[i][tk];
                    let mut qc = vec![0f32; dimn];
                    let mut qn = vec![0f32; heads.len()];
                    for (hi2, &hh) in heads.iter().enumerate() {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        unrope(qf, base, (tk + qoff) as f64, &freq, &mut qb);
                        qc[base..base + HEAD_DIM].copy_from_slice(&qb);
                        qn[hi2] = qb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    }
                    for di in 0..n_defs {
                        let mut s = 0f64;
                        for (hi2, &hh) in heads.iter().enumerate() {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            let mut cmax = f32::MIN;
                            for tok in &def_content[di] {
                                let (mut dot, mut kn) = (0f32, 0f32);
                                for j in 0..HEAD_DIM {
                                    let k = tok[base + j];
                                    dot += qc[base + j] * k;
                                    kn += k * k;
                                }
                                let cos = dot / (qn[hi2] * kn.sqrt().max(1e-6));
                                if cos > cmax {
                                    cmax = cos;
                                }
                            }
                            s += cmax as f64;
                        }
                        score[di] += s;
                    }
                }
                let rank = 1 + score.iter().filter(|&&x| x > score[target]).count();
                let famr = (0..n_defs)
                    .filter(|&d| stem(&def_ktok[d].0) == tstem)
                    .map(|fd| 1 + score.iter().filter(|&&x| x > score[fd]).count())
                    .min()
                    .unwrap_or(n_defs);
                (qoff, rank, famr)
            })
            .collect();

        let mut best = sweep.clone();
        best.sort_by_key(|r| r.1);
        println!("\n══ §72 — RoPE un-rotation QOFF sweep (case {i}, tool={truth}, theta={theta:.0}, {} call tokens) ══", calltoks.len());
        println!("  best 12 QOFF by target rank (of {n_defs} defs):");
        for (qoff, rank, famr) in best.iter().take(12) {
            println!(
                "    QOFF {:>5} → target rank {:>3}   family rank {:>3}",
                qoff, rank, famr
            );
        }
        let med = best[best.len() / 2];
        println!(
            "  worst QOFF rank {} ; median QOFF rank {} ; scrambled baseline was ~80-93",
            best.last().unwrap().1,
            med.1
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §73 — holdout Q·Q retrieval: does a decode's query find same-tool decodes?
    //
    //  The real product signal (decode→decode, same-domain — NOT the call→def wall of
    //  §21–§72). Pool each case's content decode-token Q (per head), hold each out, rank
    //  all others by per-head cosine. Does a same-tool / same-family decode land in
    //  Top-1/Top-5? Knobs: mean-subtraction (remove the common query component), layer
    //  band, raw-vs-sign (BDP). Run `S21_ONLY=1 S73=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S73").is_ok() {
        use rayon::prelude::*;
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let dimn = (0..n_cases)
            .find_map(|c| tool_q_float[c].first().map(|v| v.len()))
            .unwrap_or(0);
        // per-case pooled query over content decode tokens (non-zero Q).
        let mut case_q: Vec<Vec<f32>> = Vec::new();
        let mut case_tool: Vec<String> = Vec::new();
        for ci in 0..n_cases {
            let content: Vec<usize> = tool_ranges[ci][3]
                .iter()
                .copied()
                .filter(|&tk| {
                    tk < tool_q_float[ci].len()
                        && tool_q_float[ci][tk].iter().map(|x| x.abs()).sum::<f32>() > 1e-3
                })
                .collect();
            if content.is_empty() {
                continue;
            }
            let mut v = vec![0f32; dimn];
            for &tk in &content {
                let qf = &tool_q_float[ci][tk];
                for d in 0..dimn {
                    v[d] += qf[d];
                }
            }
            let inv = 1.0 / content.len() as f32;
            for d in 0..dimn {
                v[d] *= inv;
            }
            case_q.push(v);
            case_tool.push(tool_phase_tool[ci].clone());
        }
        let nc = case_q.len();
        // global mean query (the "common component").
        let mut gmean = vec![0f32; dimn];
        for v in &case_q {
            for d in 0..dimn {
                gmean[d] += v[d];
            }
        }
        for d in 0..dimn {
            gmean[d] /= nc as f32;
        }
        let adj_sub: Vec<Vec<f32>> = case_q
            .iter()
            .map(|v| v.iter().zip(&gmean).map(|(a, m)| a - m).collect())
            .collect();

        let n_with_sib = (0..nc)
            .filter(|&a| (0..nc).any(|b| b != a && case_tool[b] == case_tool[a]))
            .count();
        println!("\n══ §73 — holdout Q·Q same-tool retrieval ({nc} decodes, {n_with_sib} with a same-tool sibling) ══");
        println!(
            "  {:<24} {:>7} {:>7} {:>7} {:>7}",
            "config", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );

        let run = |label: &str, adj: &Vec<Vec<f32>>, lo: usize, hi: usize, sign: bool| {
            let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
            let norms: Vec<Vec<f32>> = adj
                .par_iter()
                .map(|v| {
                    heads
                        .iter()
                        .map(|&hh| {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            (0..HEAD_DIM)
                                .map(|j| v[base + j] * v[base + j])
                                .sum::<f32>()
                                .sqrt()
                                .max(1e-6)
                        })
                        .collect()
                })
                .collect();
            let res: Vec<[bool; 4]> = (0..nc)
                .into_par_iter()
                .filter_map(|a| {
                    if !(0..nc).any(|b| b != a && case_tool[b] == case_tool[a]) {
                        return None;
                    }
                    let mut sims: Vec<(usize, f64)> = (0..nc)
                        .filter(|&b| b != a)
                        .map(|b| {
                            let mut s = 0f64;
                            for (hi2, &hh) in heads.iter().enumerate() {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                if sign {
                                    let mut ag = 0i32;
                                    for j in 0..HEAD_DIM {
                                        ag += if (adj[a][base + j] >= 0.0)
                                            == (adj[b][base + j] >= 0.0)
                                        {
                                            1
                                        } else {
                                            -1
                                        };
                                    }
                                    s += ag as f64 / HEAD_DIM as f64;
                                } else {
                                    let mut dot = 0f32;
                                    for j in 0..HEAD_DIM {
                                        dot += adj[a][base + j] * adj[b][base + j];
                                    }
                                    s += (dot / (norms[a][hi2] * norms[b][hi2])) as f64;
                                }
                            }
                            (b, s / heads.len() as f64)
                        })
                        .collect();
                    sims.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
                    let same_t = |b: usize| case_tool[b] == case_tool[a];
                    let same_f = |b: usize| stem(&case_tool[b]) == stem(&case_tool[a]);
                    Some([
                        same_t(sims[0].0),
                        sims.iter().take(5).any(|&(b, _)| same_t(b)),
                        same_f(sims[0].0),
                        sims.iter().take(5).any(|&(b, _)| same_f(b)),
                    ])
                })
                .collect();
            let n = res.len().max(1);
            let pc = |k: usize| 100.0 * res.iter().filter(|r| r[k]).count() as f64 / n as f64;
            println!(
                "  {:<24} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                label,
                pc(0),
                pc(1),
                pc(2),
                pc(3)
            );
        };

        run("raw cosine all-band", &case_q, 0, 48, false);
        run("mean-sub all-band", &adj_sub, 0, 48, false);
        run("mean-sub mid-band", &adj_sub, 24, 40, false);
        run("mean-sub late-band", &adj_sub, 36, 48, false);
        run("raw mid-band", &case_q, 24, 40, false);
        run("mean-sub SIGN all-band", &adj_sub, 0, 48, true);
        run("mean-sub SIGN mid-band", &adj_sub, 24, 40, true);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §74 — per-token Q·Q retrieval with cross-token aggregation algorithms.
    //
    //  §73 pooled each decode to one mean query (blurs the distinctive name token).
    //  Here every probe (draft) decode token is scored against every stored decode token,
    //  then aggregated to a per-case score by several algorithms: maxpair (max signal),
    //  meanpair, bestprobe (Σ per-probe best), vote (consensus argmax), consec (longest
    //  consecutive run). Same-tool/family Top-1/5. Run `S21_ONLY=1 S74=1` (`S74_BAND=all`,`S74_SIGN`).
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S74").is_ok() {
        use rayon::prelude::*;
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        // S74_BAND: "all" | "lo-hi" (e.g. "24-40") | default mid 24-40.
        let band = std::env::var("S74_BAND").unwrap_or_default();
        let (lo, hi) = if band == "all" {
            (0usize, 48usize)
        } else if let Some((a, b)) = band.split_once('-') {
            (a.parse().unwrap_or(24), b.parse().unwrap_or(40))
        } else {
            (24, 40)
        };
        let heads: Vec<usize> = (lo * N_KV_HEAD..hi * N_KV_HEAD).collect();
        let sign = std::env::var("S74_SIGN").is_ok();
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let case_tool: Vec<String> = (0..n_cases).map(|c| tool_phase_tool[c].clone()).collect();
        // flatten non-zero decode tokens.
        let mut toks: Vec<(usize, usize)> = Vec::new();
        for ci in 0..n_cases {
            for &tk in &tool_ranges[ci][3] {
                if tk < tool_q_float[ci].len()
                    && tool_q_float[ci][tk].iter().map(|x| x.abs()).sum::<f32>() > 1e-3
                {
                    toks.push((ci, tk));
                }
            }
        }
        let tnorm: Vec<Vec<f32>> = toks
            .par_iter()
            .map(|&(ci, tk)| {
                let qf = &tool_q_float[ci][tk];
                heads
                    .iter()
                    .map(|&hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM)
                            .map(|j| qf[base + j] * qf[base + j])
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-6)
                    })
                    .collect()
            })
            .collect();
        let mut case_toks: Vec<Vec<usize>> = vec![Vec::new(); n_cases];
        for (i, &(ci, _)) in toks.iter().enumerate() {
            case_toks[ci].push(i);
        }

        let res: Vec<[[bool; 4]; 5]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                if case_toks[a].is_empty()
                    || !(0..n_cases).any(|b| b != a && case_tool[b] == case_tool[a])
                {
                    return None;
                }
                let mut maxpair = vec![f64::MIN; n_cases];
                let mut sumpair = vec![0f64; n_cases];
                let mut cntpair = vec![0u32; n_cases];
                let mut bestprobe = vec![0f64; n_cases];
                let mut votes = vec![0u32; n_cases];
                let mut pbc: Vec<usize> = Vec::new();
                for &pi in &case_toks[a] {
                    let (ca, ta) = toks[pi];
                    let qa = &tool_q_float[ca][ta];
                    let mut perb = vec![f64::MIN; n_cases];
                    let (mut gb, mut gc) = (f64::MIN, usize::MAX);
                    for b in 0..n_cases {
                        if b == a {
                            continue;
                        }
                        for &s in &case_toks[b] {
                            let (cb, tb2) = toks[s];
                            let qb = &tool_q_float[cb][tb2];
                            let mut sm = 0f64;
                            for (hi2, &hh) in heads.iter().enumerate() {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                if sign {
                                    let mut ag = 0i32;
                                    for j in 0..HEAD_DIM {
                                        ag += if (qa[base + j] >= 0.0) == (qb[base + j] >= 0.0) {
                                            1
                                        } else {
                                            -1
                                        };
                                    }
                                    sm += ag as f64 / HEAD_DIM as f64;
                                } else {
                                    let mut dot = 0f32;
                                    for j in 0..HEAD_DIM {
                                        dot += qa[base + j] * qb[base + j];
                                    }
                                    sm += (dot / (tnorm[pi][hi2] * tnorm[s][hi2])) as f64;
                                }
                            }
                            sm /= heads.len() as f64;
                            if sm > maxpair[b] {
                                maxpair[b] = sm;
                            }
                            sumpair[b] += sm;
                            cntpair[b] += 1;
                            if sm > perb[b] {
                                perb[b] = sm;
                            }
                            if sm > gb {
                                gb = sm;
                                gc = b;
                            }
                        }
                    }
                    for b in 0..n_cases {
                        if perb[b] > f64::MIN {
                            bestprobe[b] += perb[b];
                        }
                    }
                    if gc != usize::MAX {
                        votes[gc] += 1;
                        pbc.push(gc);
                    }
                }
                let mut consec = vec![0u32; n_cases];
                let mut run = 0u32;
                for w in 0..pbc.len() {
                    if w > 0 && pbc[w] == pbc[w - 1] {
                        run += 1;
                    } else {
                        run = 1;
                    }
                    if run > consec[pbc[w]] {
                        consec[pbc[w]] = run;
                    }
                }
                let meanpair: Vec<f64> = (0..n_cases)
                    .map(|b| {
                        if cntpair[b] > 0 {
                            sumpair[b] / cntpair[b] as f64
                        } else {
                            f64::MIN
                        }
                    })
                    .collect();
                let scores = [
                    maxpair,
                    meanpair,
                    bestprobe,
                    votes.iter().map(|&v| v as f64).collect(),
                    consec.iter().map(|&v| v as f64).collect(),
                ];
                let mut out = [[false; 4]; 5];
                for (ai, sc) in scores.iter().enumerate() {
                    let mut order: Vec<usize> = (0..n_cases).filter(|&b| b != a).collect();
                    order.sort_by(|&x, &y| sc[y].partial_cmp(&sc[x]).unwrap());
                    let st = |b: usize| case_tool[b] == case_tool[a];
                    let sf = |b: usize| stem(&case_tool[b]) == stem(&case_tool[a]);
                    out[ai] = [
                        st(order[0]),
                        order.iter().take(5).any(|&b| st(b)),
                        sf(order[0]),
                        order.iter().take(5).any(|&b| sf(b)),
                    ];
                }
                Some(out)
            })
            .collect();

        let n = res.len().max(1);
        let algos = ["maxpair", "meanpair", "bestprobe", "vote", "consec"];
        println!(
            "\n══ §74 — per-token Q·Q aggregation ({n} holdouts, band={}, {}) ══",
            if band == "all" { "all" } else { "mid" },
            if sign { "SIGN" } else { "cosine" }
        );
        println!(
            "  {:<11} {:>7} {:>7} {:>7} {:>7}",
            "algo", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        for (ai, name) in algos.iter().enumerate() {
            let pc = |k: usize| 100.0 * res.iter().filter(|r| r[ai][k]).count() as f64 / n as f64;
            println!(
                "  {:<11} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                name,
                pc(0),
                pc(1),
                pc(2),
                pc(3)
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §75 — per-token Q·Q vote with BLIND per-layer weighting.
    //
    //  §74 summed all 48 layers uniformly. Here each draft token weights each layer by how
    //  decisively that layer separates the candidates — measured blind from the layer's own
    //  similarity distribution over all stored tokens (no label). Formulas: uniform, std,
    //  zmax=(max−mean)/std, maxmean, topgap=max−2nd. Combine = Σ_L w_L·cos_L, argmax→vote.
    //  Run `S21_ONLY=1 S75=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S75").is_ok() {
        use rayon::prelude::*;
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        let nl = N_LAYERS; // 48
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let case_tool: Vec<String> = (0..n_cases).map(|c| tool_phase_tool[c].clone()).collect();
        let mut toks: Vec<(usize, usize)> = Vec::new();
        for ci in 0..n_cases {
            for &tk in &tool_ranges[ci][3] {
                if tk < tool_q_float[ci].len()
                    && tool_q_float[ci][tk].iter().map(|x| x.abs()).sum::<f32>() > 1e-3
                {
                    toks.push((ci, tk));
                }
            }
        }
        let n_heads = nl * N_KV_HEAD;
        let tnorm: Vec<Vec<f32>> = toks
            .par_iter()
            .map(|&(ci, tk)| {
                let qf = &tool_q_float[ci][tk];
                (0..n_heads)
                    .map(|hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM)
                            .map(|j| qf[base + j] * qf[base + j])
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-6)
                    })
                    .collect()
            })
            .collect();
        let mut case_toks: Vec<Vec<usize>> = vec![Vec::new(); n_cases];
        for (i, &(ci, _)) in toks.iter().enumerate() {
            case_toks[ci].push(i);
        }
        let sign = std::env::var("S75_SIGN").is_ok();
        let wnames = ["uniform", "std", "zmax", "maxmean", "topgap"];

        let res: Vec<[[bool; 4]; 5]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                if case_toks[a].is_empty()
                    || !(0..n_cases).any(|b| b != a && case_tool[b] == case_tool[a])
                {
                    return None;
                }
                let stored: Vec<usize> = (0..n_cases)
                    .filter(|&b| b != a)
                    .flat_map(|b| case_toks[b].iter().copied())
                    .collect();
                let nst = stored.len();
                let mut votes = vec![[0u32; 5]; n_cases];
                for &pi in &case_toks[a] {
                    let (ca, ta) = toks[pi];
                    let qa = &tool_q_float[ca][ta];
                    // per-stored, per-layer cosine.
                    let mut mat = vec![[0f32; 64]; nst]; // 64 ≥ nl
                    for (r, &s) in stored.iter().enumerate() {
                        let (cb, tb2) = toks[s];
                        let qb = &tool_q_float[cb][tb2];
                        for layer in 0..nl {
                            let mut c = 0f32;
                            for kvh in 0..N_KV_HEAD {
                                let hh = layer * N_KV_HEAD + kvh;
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                if sign {
                                    let mut ag = 0i32;
                                    for j in 0..HEAD_DIM {
                                        ag += if (qa[base + j] >= 0.0) == (qb[base + j] >= 0.0) {
                                            1
                                        } else {
                                            -1
                                        };
                                    }
                                    c += ag as f32 / HEAD_DIM as f32;
                                } else {
                                    let mut dot = 0f32;
                                    for j in 0..HEAD_DIM {
                                        dot += qa[base + j] * qb[base + j];
                                    }
                                    c += dot / (tnorm[pi][hh] * tnorm[s][hh]);
                                }
                            }
                            mat[r][layer] = c / N_KV_HEAD as f32;
                        }
                    }
                    // per-layer stats over candidates.
                    let mut w = [[0f32; 64]; 5];
                    for layer in 0..nl {
                        let (mut sum, mut sq, mut mx, mut m2) = (0f32, 0f32, f32::MIN, f32::MIN);
                        for r in 0..nst {
                            let v = mat[r][layer];
                            sum += v;
                            sq += v * v;
                            if v > mx {
                                m2 = mx;
                                mx = v;
                            } else if v > m2 {
                                m2 = v;
                            }
                        }
                        let mean = sum / nst as f32;
                        let std = (sq / nst as f32 - mean * mean).max(0.0).sqrt().max(1e-4);
                        w[0][layer] = 1.0;
                        w[1][layer] = std;
                        w[2][layer] = ((mx - mean) / std).max(0.0);
                        w[3][layer] = (mx - mean).max(0.0);
                        w[4][layer] = (mx - m2).max(0.0);
                    }
                    for f in 0..5 {
                        let (mut best, mut bc) = (f32::MIN, usize::MAX);
                        for r in 0..nst {
                            let mut sc = 0f32;
                            for layer in 0..nl {
                                sc += w[f][layer] * mat[r][layer];
                            }
                            if sc > best {
                                best = sc;
                                bc = toks[stored[r]].0;
                            }
                        }
                        if bc != usize::MAX {
                            votes[bc][f] += 1;
                        }
                    }
                }
                let mut out = [[false; 4]; 5];
                for f in 0..5 {
                    let mut order: Vec<usize> = (0..n_cases).filter(|&b| b != a).collect();
                    order.sort_by(|&x, &y| votes[y][f].cmp(&votes[x][f]));
                    let st = |b: usize| case_tool[b] == case_tool[a];
                    let sf = |b: usize| stem(&case_tool[b]) == stem(&case_tool[a]);
                    out[f] = [
                        st(order[0]),
                        order.iter().take(5).any(|&b| st(b)),
                        sf(order[0]),
                        order.iter().take(5).any(|&b| sf(b)),
                    ];
                }
                Some(out)
            })
            .collect();

        let n = res.len().max(1);
        println!(
            "\n══ §75 — per-token vote, blind per-layer weighting ({n} holdouts, all-48, {}) ══",
            if sign { "SIGN" } else { "cosine" }
        );
        println!(
            "  {:<10} {:>7} {:>7} {:>7} {:>7}",
            "weight", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        for (f, name) in wnames.iter().enumerate() {
            let pc = |k: usize| 100.0 * res.iter().filter(|r| r[f][k]).count() as f64 / n as f64;
            println!(
                "  {:<10} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                name,
                pc(0),
                pc(1),
                pc(2),
                pc(3)
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §76 — family / common-hit cross-scoring on the per-token vote (§47–§50 idea).
    //
    //  The family wall: a draft token whose best match is a family-shared hit votes for an
    //  arbitrary sibling. Two BLIND cross-scoring levers on each token's per-case score:
    //   null   — subtract candidate promiscuity (mean blind case-case sim to everything),
    //   famres — subtract candidate's blind-family mean (top-m most-similar cases),
    //            equalising the shared bulk so only the distinctive residual votes.
    //  Families are detected by pooled-query cosine (NOT the stem label). Run `S21_ONLY=1 S76=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S76").is_ok() {
        use rayon::prelude::*;
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }
        let heads: Vec<usize> = (0..N_LAYERS * N_KV_HEAD).collect();
        let m: usize = std::env::var("S76_M")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        let n_cases = tool_phase_tool
            .len()
            .min(tool_q_float.len())
            .min(tool_ranges.len());
        let case_tool: Vec<String> = (0..n_cases).map(|c| tool_phase_tool[c].clone()).collect();
        let dimn = (0..n_cases)
            .find_map(|c| tool_q_float[c].first().map(|v| v.len()))
            .unwrap_or(0);
        let mut toks: Vec<(usize, usize)> = Vec::new();
        for ci in 0..n_cases {
            for &tk in &tool_ranges[ci][3] {
                if tk < tool_q_float[ci].len()
                    && tool_q_float[ci][tk].iter().map(|x| x.abs()).sum::<f32>() > 1e-3
                {
                    toks.push((ci, tk));
                }
            }
        }
        let tnorm: Vec<Vec<f32>> = toks
            .par_iter()
            .map(|&(ci, tk)| {
                let qf = &tool_q_float[ci][tk];
                heads
                    .iter()
                    .map(|&hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM)
                            .map(|j| qf[base + j] * qf[base + j])
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-6)
                    })
                    .collect()
            })
            .collect();
        let mut case_toks: Vec<Vec<usize>> = vec![Vec::new(); n_cases];
        for (i, &(ci, _)) in toks.iter().enumerate() {
            case_toks[ci].push(i);
        }

        // pooled case query → blind case-case similarity → promiscuity + families.
        let cq: Vec<Vec<f32>> = (0..n_cases)
            .map(|ci| {
                let content: Vec<usize> = tool_ranges[ci][3]
                    .iter()
                    .copied()
                    .filter(|&tk| {
                        tk < tool_q_float[ci].len()
                            && tool_q_float[ci][tk].iter().map(|x| x.abs()).sum::<f32>() > 1e-3
                    })
                    .collect();
                let mut v = vec![0f32; dimn];
                if !content.is_empty() {
                    for &tk in &content {
                        for d in 0..dimn {
                            v[d] += tool_q_float[ci][tk][d];
                        }
                    }
                    let inv = 1.0 / content.len() as f32;
                    for d in 0..dimn {
                        v[d] *= inv;
                    }
                }
                v
            })
            .collect();
        let cqn: Vec<Vec<f32>> = cq
            .par_iter()
            .map(|v| {
                heads
                    .iter()
                    .map(|&hh| {
                        let base = (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                        (0..HEAD_DIM)
                            .map(|j| v[base + j] * v[base + j])
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-6)
                    })
                    .collect()
            })
            .collect();
        let casesim: Vec<Vec<f32>> = (0..n_cases)
            .into_par_iter()
            .map(|b| {
                (0..n_cases)
                    .map(|c| {
                        if b == c {
                            return 0f32;
                        }
                        let mut s = 0f32;
                        for (hi, &hh) in heads.iter().enumerate() {
                            let base =
                                (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                            let mut dot = 0f32;
                            for j in 0..HEAD_DIM {
                                dot += cq[b][base + j] * cq[c][base + j];
                            }
                            s += dot / (cqn[b][hi] * cqn[c][hi]);
                        }
                        s / heads.len() as f32
                    })
                    .collect()
            })
            .collect();
        let promisc: Vec<f32> = (0..n_cases)
            .map(|b| {
                let v: f32 = (0..n_cases)
                    .filter(|&c| c != b)
                    .map(|c| casesim[b][c])
                    .sum();
                v / (n_cases - 1).max(1) as f32
            })
            .collect();
        let (pmean, pstd) = {
            let mu = promisc.iter().sum::<f32>() / n_cases as f32;
            let sd = (promisc.iter().map(|x| (x - mu) * (x - mu)).sum::<f32>() / n_cases as f32)
                .sqrt()
                .max(1e-6);
            (mu, sd)
        };
        let promz: Vec<f32> = promisc.iter().map(|p| (p - pmean) / pstd).collect();
        let fam: Vec<Vec<usize>> = (0..n_cases)
            .map(|b| {
                let mut idx: Vec<usize> = (0..n_cases).filter(|&c| c != b).collect();
                idx.sort_by(|&x, &y| casesim[b][y].partial_cmp(&casesim[b][x]).unwrap());
                idx.truncate(m);
                idx
            })
            .collect();

        let variants = ["vote", "null", "famres", "both"];
        let res: Vec<[[bool; 4]; 4]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                if case_toks[a].is_empty()
                    || !(0..n_cases).any(|b| b != a && case_tool[b] == case_tool[a])
                {
                    return None;
                }
                let mut votes = vec![[0u32; 4]; n_cases];
                for &pi in &case_toks[a] {
                    let (ca, ta) = toks[pi];
                    let qa = &tool_q_float[ca][ta];
                    let mut cs = vec![f32::MIN; n_cases];
                    for b in 0..n_cases {
                        if b == a {
                            continue;
                        }
                        for &s in &case_toks[b] {
                            let (cb, tb2) = toks[s];
                            let qb = &tool_q_float[cb][tb2];
                            let mut c = 0f32;
                            for (hi, &hh) in heads.iter().enumerate() {
                                let base =
                                    (hh / N_KV_HEAD) * PER_LAYER_DIM + (hh % N_KV_HEAD) * HEAD_DIM;
                                let mut dot = 0f32;
                                for j in 0..HEAD_DIM {
                                    dot += qa[base + j] * qb[base + j];
                                }
                                c += dot / (tnorm[pi][hi] * tnorm[s][hi]);
                            }
                            c /= heads.len() as f32;
                            if c > cs[b] {
                                cs[b] = c;
                            }
                        }
                    }
                    // z-score cs over valid candidates (for the null term's scale).
                    let valid: Vec<usize> = (0..n_cases).filter(|&b| cs[b] > f32::MIN).collect();
                    let mu = valid.iter().map(|&b| cs[b]).sum::<f32>() / valid.len().max(1) as f32;
                    let sd = (valid
                        .iter()
                        .map(|&b| (cs[b] - mu) * (cs[b] - mu))
                        .sum::<f32>()
                        / valid.len().max(1) as f32)
                        .sqrt()
                        .max(1e-6);
                    let fammean = |sc: &[f32], b: usize| {
                        let f: Vec<f32> = fam[b]
                            .iter()
                            .filter(|&&c| c != a && sc[c] > f32::MIN)
                            .map(|&c| sc[c])
                            .collect();
                        if f.is_empty() {
                            0.0
                        } else {
                            f.iter().sum::<f32>() / f.len() as f32
                        }
                    };
                    // build the four adjusted score vectors.
                    let s_vote = cs.clone();
                    let s_null: Vec<f32> = (0..n_cases)
                        .map(|b| {
                            if cs[b] > f32::MIN {
                                (cs[b] - mu) / sd - promz[b]
                            } else {
                                f32::MIN
                            }
                        })
                        .collect();
                    let s_famres: Vec<f32> = (0..n_cases)
                        .map(|b| {
                            if cs[b] > f32::MIN {
                                cs[b] - fammean(&cs, b)
                            } else {
                                f32::MIN
                            }
                        })
                        .collect();
                    let s_both: Vec<f32> = (0..n_cases)
                        .map(|b| {
                            if s_null[b] > f32::MIN {
                                s_null[b] - fammean(&s_null, b)
                            } else {
                                f32::MIN
                            }
                        })
                        .collect();
                    for (vi, sc) in [&s_vote, &s_null, &s_famres, &s_both].iter().enumerate() {
                        let bc = (0..n_cases)
                            .filter(|&b| b != a)
                            .max_by(|&x, &y| sc[x].partial_cmp(&sc[y]).unwrap())
                            .unwrap();
                        votes[bc][vi] += 1;
                    }
                }
                let mut out = [[false; 4]; 4];
                for vi in 0..4 {
                    let mut order: Vec<usize> = (0..n_cases).filter(|&b| b != a).collect();
                    order.sort_by(|&x, &y| votes[y][vi].cmp(&votes[x][vi]));
                    let st = |b: usize| case_tool[b] == case_tool[a];
                    let sf = |b: usize| stem(&case_tool[b]) == stem(&case_tool[a]);
                    out[vi] = [
                        st(order[0]),
                        order.iter().take(5).any(|&b| st(b)),
                        sf(order[0]),
                        order.iter().take(5).any(|&b| sf(b)),
                    ];
                }
                Some(out)
            })
            .collect();

        let n = res.len().max(1);
        println!("\n══ §76 — family/common-hit cross-scoring on the vote ({n} holdouts, all-48 cosine, m={m}) ══");
        println!(
            "  {:<9} {:>7} {:>7} {:>7} {:>7}",
            "variant", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        for (vi, name) in variants.iter().enumerate() {
            let pc = |k: usize| 100.0 * res.iter().filter(|r| r[vi][k]).count() as f64 / n as f64;
            println!(
                "  {:<9} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                name,
                pc(0),
                pc(1),
                pc(2),
                pc(3)
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §77 — Q·Q consensus vote from the PERSISTED wide-Q signatures (the
    //  substrate-native product path).
    //
    //  §73–§76 reconstruct per-token float Q from the turns' R16 chunks. But under
    //  production KV compression only ~36% of chunks stay R16, so that reconstruction
    //  now samples a third of the tokens. The substrate persists `wide_q_sigs`: the
    //  complete per-token `sign(Q)` history (every head of every layer), captured
    //  live at seal *before* compression. §77 now scores the PRODUCTION UNIT — one
    //  probe per PROJECTION EVENT: its lookup signature is the rolling-`ROLLING_BACK`
    //  wide-Q window ending at the projection point (exactly what a production
    //  reprojection keys on), and its tool label is the `tools` section that
    //  projection locked. It runs the §74 winning recipe — uniform per-token
    //  consensus vote, SIGN/BDP — over those windows, leaving the probe's whole
    //  CONVERSATION out of the gallery. This proves the product path works from
    //  exactly what the substrate stores, and validates the projection→section links
    //  end to end. Run `S21_ONLY=1 S77=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S77").is_ok() {
        use candle_conversation::provenance::WideQSig;
        use rayon::prelude::*;
        use std::collections::HashSet;

        // Family stem: `telnet_session_list` → `session_list` (same as §74–§76).
        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }

        // Each probe is one projection event: its lookup signature is the rolling
        // wide-Q window ending at the projection point, its tool label is the
        // `tools` section that projection locked (the substrate-native ground
        // truth), and its conversation is the turn it fired in. Leave-one-conver-
        // sation-out below holds out every projection of the probe's own turn so a
        // turn's other (near-adjacent) reprojection windows can't leak.
        let cases = projection_probe_cases(&substrate);
        let case_tool: Vec<String> = cases.iter().map(|c| c.tool.clone()).collect();
        let case_conv: Vec<u64> = cases.iter().map(|c| c.conv).collect();
        let case_win: Vec<Vec<WideQSig>> = cases.iter().map(|c| c.window.clone()).collect();
        let n_cases = case_tool.len();
        let n_convs = case_conv.iter().collect::<HashSet<_>>().len();

        // Flatten every case's window tokens into one array for the pairwise scan.
        let mut flat_case: Vec<usize> = Vec::new();
        let mut flat_words: Vec<Vec<u64>> = Vec::new();
        for (ci, win) in case_win.iter().enumerate() {
            for tok in win {
                flat_case.push(ci);
                flat_words.push(tok.words.clone());
            }
        }
        let words_per_tok = flat_words.first().map(|w| w.len()).unwrap_or(0);
        eprintln!(
            "§77: {} projection probes over {} conversations; rolling-{} lookback; {} window tokens, {} words/token",
            n_cases, n_convs, ROLLING_BACK, flat_words.len(), words_per_tok
        );

        // §74 winning recipe: each draft token votes for the stored token it best
        // matches (per-token sign agreement = XOR+popcount over all head words — a raw
        // popcount argmax is identical to §74's ±1/head sum since the bit total is
        // constant). Tally votes per case, fold to tool, leave-one-case-out ranked.
        let res: Vec<[bool; 4]> = (0..n_cases)
            .into_par_iter()
            .filter_map(|a| {
                if !(0..n_cases)
                    .any(|b| case_conv[b] != case_conv[a] && case_tool[b] == case_tool[a])
                {
                    return None; // no same-tool sibling in another conversation — unscorable
                }
                let mine: Vec<usize> = (0..flat_words.len())
                    .filter(|&j| flat_case[j] == a)
                    .collect();
                if mine.is_empty() {
                    return None;
                }
                let mut votes = vec![0u32; n_cases];
                for &qi in &mine {
                    let wa = &flat_words[qi];
                    let (mut best_ag, mut bc) = (0u32, usize::MAX);
                    for j in 0..flat_words.len() {
                        if case_conv[flat_case[j]] == case_conv[a] {
                            continue; // hold the whole query conversation out
                        }
                        let wb = &flat_words[j];
                        if wb.len() != wa.len() {
                            continue;
                        }
                        let mut ag = 0u32;
                        for w in 0..wa.len() {
                            ag += (!(wa[w] ^ wb[w])).count_ones();
                        }
                        if ag > best_ag {
                            best_ag = ag;
                            bc = flat_case[j];
                        }
                    }
                    if bc != usize::MAX {
                        votes[bc] += 1;
                    }
                }
                let mut order: Vec<usize> = (0..n_cases)
                    .filter(|&b| case_conv[b] != case_conv[a])
                    .collect();
                order.sort_by(|&x, &y| votes[y].cmp(&votes[x]));
                let st = |b: usize| case_tool[b] == case_tool[a];
                let sf = |b: usize| stem(&case_tool[b]) == stem(&case_tool[a]);
                Some([
                    st(order[0]),
                    order.iter().take(5).any(|&b| st(b)),
                    sf(order[0]),
                    order.iter().take(5).any(|&b| sf(b)),
                ])
            })
            .collect();

        let n = res.len().max(1);
        println!(
            "\n══ §77 — Q·Q vote from PERSISTED wide-Q sigs ({n} scorable holdouts of {n_cases} cases, all-48 SIGN) ══"
        );
        println!(
            "  {:<22} {:>7} {:>7} {:>7} {:>7}",
            "source", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );
        let pc = |k: usize| 100.0 * res.iter().filter(|r| r[k]).count() as f64 / n as f64;
        println!(
            "  {:<22} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            "persisted wide-Q",
            pc(0),
            pc(1),
            pc(2),
            pc(3)
        );
        println!(
            "  {:<22} {:>7} {:>7} {:>7} {:>7}   (doc §22, R16 reconstruction)",
            "§74 SIGN all-48 (ref)", "50.3", "68.1", "65.9", "80.0"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  §78 — k-fold Q·Q vote from the persisted wide-Q signatures.
    //
    //  §77 was leave-one-conversation-out: a probe's corpus still held all 3 other
    //  calls of its own tool. §78 makes probe and gallery DISJOINT by fold — each
    //  tool's CONVERSATIONS are round-robin'd across k folds; a probe projection is
    //  scored only against conversations in OTHER folds, so its fold-mates (including
    //  same-tool ones) are held out of the gallery. Sweeping k shrinks the per-tool
    //  gallery (k=4 → 3 stored siblings, k=2 → 2), showing how retrieval degrades as
    //  fewer past calls are stored. Same probe unit and recipe as §77 (per-projection
    //  rolling wide-Q window, uniform per-token vote, SIGN); tool label = the
    //  projection event's selected section. Run `S21_ONLY=1 S78=1`.
    // ════════════════════════════════════════════════════════════════════════
    if std::env::var("S78").is_ok() {
        use candle_conversation::provenance::WideQSig;
        use rayon::prelude::*;
        use std::collections::HashSet;

        fn stem(n: &str) -> String {
            let p: Vec<&str> = n.rsplitn(3, '_').collect();
            if p.len() >= 2 {
                format!("{}_{}", p[1], p[0])
            } else {
                n.to_string()
            }
        }

        // Per-projection probe cases — same source as §77 (rolling wide-Q window
        // ending at each projection point, tool label from that projection's
        // selection). Folds are assigned per CONVERSATION, not per projection, so
        // every reprojection of a turn shares one fold and probe/gallery stay
        // disjoint by conversation.
        let cases = projection_probe_cases(&substrate);
        let case_tool: Vec<String> = cases.iter().map(|c| c.tool.clone()).collect();
        let case_conv: Vec<u64> = cases.iter().map(|c| c.conv).collect();
        let case_win: Vec<Vec<WideQSig>> = cases.iter().map(|c| c.window.clone()).collect();
        let n_cases = case_tool.len();

        // Settled tool per conversation (last projection wins), then round-robin
        // each tool's conversations across folds so folds stay tool-balanced.
        let mut conv_tool: HashMap<u64, String> = HashMap::new();
        for c in &cases {
            conv_tool.insert(c.conv, c.tool.clone());
        }
        let mut convs_by_tool: HashMap<&str, Vec<u64>> = HashMap::new();
        for (cv, t) in &conv_tool {
            convs_by_tool.entry(t.as_str()).or_default().push(*cv);
        }
        let mut conv_pos: HashMap<u64, usize> = HashMap::new();
        for cvs in convs_by_tool.values_mut() {
            cvs.sort_unstable();
            for (p, cv) in cvs.iter().enumerate() {
                conv_pos.insert(*cv, p);
            }
        }

        // Flatten window tokens for the pairwise scan.
        let mut flat_case: Vec<usize> = Vec::new();
        let mut flat_words: Vec<Vec<u64>> = Vec::new();
        for (ci, win) in case_win.iter().enumerate() {
            for tok in win {
                flat_case.push(ci);
                flat_words.push(tok.words.clone());
            }
        }
        let n_tools = case_tool.iter().collect::<HashSet<_>>().len();
        eprintln!(
            "§78: {} projection probes over {} conversations, {} window tokens, {} tools; rolling-{} lookback",
            n_cases,
            conv_tool.len(),
            flat_words.len(),
            n_tools,
            ROLLING_BACK
        );

        let ks: Vec<usize> = std::env::var("S78_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(|k| vec![k])
            .unwrap_or_else(|| vec![4usize, 3, 2]);

        println!(
            "\n══ §78 — k-fold Q·Q vote from persisted wide-Q sigs (probe/gallery disjoint by fold) ══"
        );
        println!(
            "  {:<20} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7}",
            "config", "n", "gsib", "Tool-1", "Tool-5", "Fam-1", "Fam-5"
        );

        for &k in &ks {
            // fold[ci] = its conversation's round-robin position within tool, mod k.
            let fold: Vec<usize> = (0..n_cases)
                .map(|ci| conv_pos[&case_conv[ci]] % k)
                .collect();

            // Per-probe result + its same-tool gallery-sibling count (for reporting).
            let out: Vec<([bool; 4], usize)> = (0..n_cases)
                .into_par_iter()
                .filter_map(|a| {
                    let fa = fold[a];
                    let gsib = (0..n_cases)
                        .filter(|&b| b != a && fold[b] != fa && case_tool[b] == case_tool[a])
                        .count();
                    if gsib == 0 {
                        return None; // no same-tool call in the gallery — unscorable
                    }
                    let mine: Vec<usize> = (0..flat_words.len())
                        .filter(|&j| flat_case[j] == a)
                        .collect();
                    if mine.is_empty() {
                        return None;
                    }
                    let mut votes = vec![0u32; n_cases];
                    for &qi in &mine {
                        let wa = &flat_words[qi];
                        let (mut best_ag, mut bc) = (0u32, usize::MAX);
                        for j in 0..flat_words.len() {
                            let cj = flat_case[j];
                            if fold[cj] == fa {
                                continue; // gallery = other folds only (excludes self + fold-mates)
                            }
                            let wb = &flat_words[j];
                            if wb.len() != wa.len() {
                                continue;
                            }
                            let mut ag = 0u32;
                            for w in 0..wa.len() {
                                ag += (!(wa[w] ^ wb[w])).count_ones();
                            }
                            if ag > best_ag {
                                best_ag = ag;
                                bc = cj;
                            }
                        }
                        if bc != usize::MAX {
                            votes[bc] += 1;
                        }
                    }
                    let mut order: Vec<usize> = (0..n_cases).filter(|&b| fold[b] != fa).collect();
                    order.sort_by(|&x, &y| votes[y].cmp(&votes[x]));
                    let st = |b: usize| case_tool[b] == case_tool[a];
                    let sf = |b: usize| stem(&case_tool[b]) == stem(&case_tool[a]);
                    let row = [
                        st(order[0]),
                        order.iter().take(5).any(|&b| st(b)),
                        sf(order[0]),
                        order.iter().take(5).any(|&b| sf(b)),
                    ];
                    Some((row, gsib))
                })
                .collect();

            let n = out.len().max(1);
            let gsib_avg = out.iter().map(|(_, g)| *g as f64).sum::<f64>() / n as f64;
            let pc =
                |kk: usize| 100.0 * out.iter().filter(|(r, _)| r[kk]).count() as f64 / n as f64;
            println!(
                "  {:<20} {:>6} {:>6.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                format!("k={k}"),
                n,
                gsib_avg,
                pc(0),
                pc(1),
                pc(2),
                pc(3)
            );
        }
        println!(
            "  {:<20} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7}   (prior whole-turn-window §77 LOO)",
            "whole-turn (ref)", "372", "3.0", "55.9", "89.0", "65.1", "93.0"
        );
    }

    Ok(())
}
