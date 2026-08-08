//! Fast, profileable microbenchmark harnesses for the paged latent-attention
//! decode/prefill kernels.
//!
//! Both harnesses synthesize a dataset shaped like a REAL deep-context
//! attention step — a 128-token sliding window placed at a high base position
//! (`[D-128, D)`), a large compressed gallery sized to `depth_tokens / ratio`
//! entries (spilled to CPU past `HOT_ENTRY_CAP`, exactly like production), and
//! a scattered top-`topk` selection into it — WITHOUT loading the 164 GB model,
//! so a run completes in seconds. Each runs a warmup then a timed launch loop
//! over the kernel and reports per-launch latency / throughput, then validates
//! one launch against a table-faithful f64 sink-softmax reference (a tolerance
//! gate: kernel-perf iteration must not silently corrupt output).
//!
//! `ncu`/`nsys` attach to the example binaries (`latent_decode_bench`,
//! `latent_prefill_bench`); filter to the `latent_*_kernel` names to skip the
//! one-time setup launches (gallery sign-pack, RoPE table build, gathers).
//!
//! The reference sources its RoPE cos/sin from the SAME device table the kernel
//! reads (`build_rope_table` → read back → factored lookup), so it stays
//! faithful with zero duplicated trig constants; it is the correctness contract
//! we hold while making the kernels extremely fast (and, next, adding PalQuant
//! window formats — the arena format is the one seam this harness leaves open).

#![cfg(feature = "cuda")]

use std::time::Instant;

use candle::{DType, Device, Result, Tensor};
use half::bf16;

use super::gallery::FloatGallery;
use super::paged::{
    build_rope_table, e4m3_to_f32, f32_to_e4m3, fp8_store_tag, paged_latent_decode_raw,
    paged_latent_prefill, CorpusCache, LatentWorkspace, SyntheticSlots, HEAD_DIM, NOPE_DIM,
    ROPE_DIM, ROPE_HI_DIM, ROPE_LO_BITS, ROPE_LO_DIM,
};
use super::rope::yarn_freqs;

/// Query heads (DeepSeek-V4-Flash config default).
const N_HEADS: usize = 64;
/// Sliding-window size (config default).
const WINDOW: usize = 128;
/// Indexer scoring-key width the gallery carries (config default). Irrelevant
/// to the attention kernels — only the BDP/gather side touches it — but real,
/// so the gallery memory footprint and gather cost match production.
const INDEX_HEAD_DIM: usize = 128;

// ─── deterministic PRNG (splitmix64; no rand / clock dependency) ──────────────

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f32 {
        ((self.bits() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.bits() % n as u64) as usize
    }
}

/// One FP8(E4M3)-exact latent element (what the window arena stores and the
/// kernel reads back through the FP8 round-trip). Returning the round-tripped
/// value means the host reference and the arena see identical bits.
fn fp8_elem(rng: &mut Rng) -> f32 {
    e4m3_to_f32(f32_to_e4m3(rng.sym()))
}
/// One bf16-exact latent element (queries / fresh latents enter as bf16).
fn bf16_elem(rng: &mut Rng) -> f32 {
    bf16::from_f32(rng.sym()).to_f32()
}

fn fp8_latent(rng: &mut Rng) -> [f32; HEAD_DIM] {
    std::array::from_fn(|_| fp8_elem(rng))
}
fn bf16_latent(rng: &mut Rng) -> [f32; HEAD_DIM] {
    std::array::from_fn(|_| bf16_elem(rng))
}

fn latents_2d(dev: &Device, rows: &[[f32; HEAD_DIM]]) -> Result<Tensor> {
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (rows.len(), HEAD_DIM), dev)?.to_dtype(DType::BF16)
}
fn latents_3d(dev: &Device, rows: &[[f32; HEAD_DIM]], d0: usize, d1: usize) -> Result<Tensor> {
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (d0, d1, HEAD_DIM), dev)?.to_dtype(DType::BF16)
}

// ─── table-faithful RoPE reference (reads the shipping device table) ──────────

/// The kernel's `rope_lookup`, on host: factored (hi, lo) position split +
/// angle-addition, over the same table `build_rope_table` produced.
fn table_lookup(tab: &[f32], pos: usize, j: usize) -> (f32, f32) {
    let nf = ROPE_DIM / 2;
    let hi = (pos >> ROPE_LO_BITS).min(ROPE_HI_DIM - 1);
    let lo = pos & (ROPE_LO_DIM - 1);
    let ih = (hi * nf + j) * 2;
    let il = ((ROPE_HI_DIM + lo) * nf + j) * 2;
    let (sh, ch) = (tab[ih], tab[ih + 1]);
    let (sl, cl) = (tab[il], tab[il + 1]);
    (sh * cl + ch * sl, ch * cl - sh * sl)
}
/// Forward interleaved-pair rotation on the rope dims at `pos`.
fn rope_fwd(v: &mut [f32; HEAD_DIM], pos: usize, tab: &[f32]) {
    for k in 0..ROPE_DIM / 2 {
        let (s, c) = table_lookup(tab, pos, k);
        let d = NOPE_DIM + 2 * k;
        let (x0, x1) = (v[d], v[d + 1]);
        v[d] = x0 * c - x1 * s;
        v[d + 1] = x0 * s + x1 * c;
    }
}
/// Inverse (de-)rotation on the rope dims at `pos`.
fn derotate(v: &mut [f32; HEAD_DIM], pos: usize, tab: &[f32]) {
    for k in 0..ROPE_DIM / 2 {
        let (s, c) = table_lookup(tab, pos, k);
        let d = NOPE_DIM + 2 * k;
        let (x0, x1) = (v[d], v[d + 1]);
        v[d] = x0 * c + x1 * s;
        v[d + 1] = x1 * c - x0 * s;
    }
}

/// f64 sink-softmax attention over POST-rope `keys` for one head, de-rotated at
/// `q_pos`. `q` is already roped. The kernel int8-quantizes K for the QK dot
/// and reads bf16-staged V for the PV; this full-precision reference therefore
/// matches only at int8/bf16 tolerance (the ~0.02·scale envelope), which is all
/// the perf gate needs — gross corruption (the failure mode of a bad kernel
/// edit) is orders of magnitude larger.
fn ref_head(
    q_roped: &[f32; HEAD_DIM],
    keys: &[[f32; HEAD_DIM]],
    sink: f32,
    scale: f32,
    q_pos: usize,
    tab: &[f32],
) -> [f32; HEAD_DIM] {
    let mut logits: Vec<f64> = keys
        .iter()
        .map(|k| {
            let mut acc = 0.0f64;
            for d in 0..HEAD_DIM {
                acc += q_roped[d] as f64 * k[d] as f64;
            }
            acc * scale as f64
        })
        .collect();
    logits.push(sink as f64);
    let m = logits.iter().cloned().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    let mut val = [0.0f64; HEAD_DIM];
    for (i, k) in keys.iter().enumerate() {
        let p = exps[i] / z;
        for d in 0..HEAD_DIM {
            val[d] += p * k[d] as f64;
        }
    }
    let mut out: [f32; HEAD_DIM] = std::array::from_fn(|d| val[d] as f32);
    derotate(&mut out, q_pos, tab);
    out
}

// ─── shared setup ─────────────────────────────────────────────────────────────

/// CSA-shaped RoPE table (θ=160000 + YaRN). The exact frequencies are
/// immaterial to both throughput (RoPE is a table read) and the gate (reference
/// reads this same table), but they are realistic.
fn csa_rope_table(dev: &Device) -> Result<Tensor> {
    let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 160_000.0, 4096, 16.0, 32.0, 1.0)
        .into_iter()
        .map(|f| f as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs_v, ROPE_DIM / 2, dev)?;
    build_rope_table(&freqs)
}

/// Fill a gallery with `g_total` entries (attn rows random f32, scoring keys
/// random f32, group-start positions `gid·ratio`), appended in batches so the
/// spill at `HOT_ENTRY_CAP` fires exactly as in production.
fn build_gallery(
    dev: &Device,
    rng: &mut Rng,
    g_total: usize,
    ratio: usize,
) -> Result<FloatGallery> {
    let mut gallery = FloatGallery::new(dev, HEAD_DIM, INDEX_HEAD_DIM, 8192)?;
    let batch = 8192usize;
    let mut done = 0usize;
    while done < g_total {
        let n = batch.min(g_total - done);
        let attn: Vec<f32> = (0..n * HEAD_DIM).map(|_| rng.sym()).collect();
        let keys: Vec<f32> = (0..n * INDEX_HEAD_DIM).map(|_| rng.sym()).collect();
        let positions: Vec<u32> = (0..n).map(|i| ((done + i) * ratio) as u32).collect();
        let attn_t = Tensor::from_vec(attn, (n, HEAD_DIM), dev)?;
        let keys_t = Tensor::from_vec(keys, (n, INDEX_HEAD_DIM), dev)?;
        gallery.append_batch(&attn_t, &keys_t, &positions)?;
        done += n;
    }
    Ok(gallery)
}

/// `topk` distinct, scattered gids in `[0, g_total)` — a strided pick with
/// per-slot jitter, so the gather touches the corpus the way a real selection
/// does (spread across the whole depth) without duplicate rows.
fn scattered_gids(rng: &mut Rng, g_total: usize, topk: usize) -> Vec<u32> {
    let stride = (g_total / topk).max(1);
    (0..topk)
        .map(|k| {
            let base = k * stride;
            let jit = if stride > 1 { rng.below(stride) } else { 0 };
            ((base + jit).min(g_total - 1)) as u32
        })
        .collect()
}

// ─── public config / report ───────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct DecodeCfg {
    /// Concurrent decode sessions in one wave (production target: 64).
    pub slots: usize,
    /// Context depth in tokens — the window sits at `[D-128, D)` and the
    /// compressed gallery holds `D/ratio` entries.
    pub depth_tokens: usize,
    /// Selected compressed entries per slot (≤ `index_topk` = 512).
    pub topk: usize,
    /// Compression ratio (CSA=4, HCA=128) — sets the gallery size `D/ratio`.
    pub ratio: usize,
    /// Split-KV factor override (0 = auto-size to fill the device). Exposed for
    /// profiling sweeps — more splits = more resident blocks to hide the
    /// latency stalls, at the cost of more combine work.
    pub splits: usize,
    pub warmup: usize,
    pub iters: usize,
    pub seed: u64,
}

impl Default for DecodeCfg {
    fn default() -> Self {
        Self {
            slots: 64,
            depth_tokens: 200_000,
            topk: 512,
            ratio: 4,
            splits: 0,
            warmup: 20,
            iters: 200,
            seed: 0x1234_5678,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PrefillCfg {
    /// New-prompt tokens absorbed in this prefill (each a fresh-diagonal query).
    pub total_q: usize,
    pub depth_tokens: usize,
    pub topk: usize,
    pub ratio: usize,
    pub warmup: usize,
    pub iters: usize,
    pub seed: u64,
}

impl Default for PrefillCfg {
    fn default() -> Self {
        Self {
            total_q: 4096,
            depth_tokens: 200_000,
            topk: 512,
            ratio: 4,
            warmup: 5,
            iters: 50,
            seed: 0x0BAD_F00D,
        }
    }
}

pub struct Report {
    pub label: String,
    pub launches: usize,
    pub keys_per_query: usize,
    pub gallery_entries: usize,
    pub spilled: bool,
    /// Wall time per kernel *call* (decode: one launch; prefill: one call =
    /// several chunked launches), including the wrapper's device sync.
    pub per_call_ms: f64,
    pub tokens_per_s: f64,
    pub keys_per_s: f64,
    /// Worst relative error over the spot-checked (slot/query, head) pairs.
    pub max_rel_err: f32,
}

impl Report {
    pub fn print(&self) {
        println!("── {} ──", self.label);
        println!(
            "  gallery={} entries ({}), keys/query={}",
            self.gallery_entries,
            if self.spilled {
                "spilled→CPU"
            } else {
                "GPU-resident"
            },
            self.keys_per_query,
        );
        println!(
            "  {:.3} ms/call over {} calls  →  {:.1} tok/s, {:.2} G key·dim/s",
            self.per_call_ms,
            self.launches,
            self.tokens_per_s,
            self.keys_per_s * HEAD_DIM as f64 / 1e9,
        );
        println!(
            "  correctness gate: max_rel_err = {:.4} (< 0.05 ✓)",
            self.max_rel_err
        );
    }
}

/// Tolerance the correctness gate holds — int8-QK + bf16-PV round-off lands
/// around 0.02·scale; anything near this bound is a real regression.
const GATE_TOL: f32 = 0.05;

// ─── decode harness ───────────────────────────────────────────────────────────

pub fn run_decode(dev: &Device, cfg: DecodeCfg) -> Result<Report> {
    if cfg.topk > 512 {
        candle::bail!("decode topk {} exceeds index_topk 512", cfg.topk);
    }
    let t_setup = Instant::now();
    let mut rng = Rng::new(cfg.seed);
    let base = cfg.depth_tokens.saturating_sub(WINDOW);
    let scale = (HEAD_DIM as f64).powf(-0.5) as f32;

    // Window arena: `slots` slots, each a 128-token window based at `D-128`.
    let windows: Vec<Vec<[f32; HEAD_DIM]>> = (0..cfg.slots)
        .map(|_| (0..WINDOW).map(|_| fp8_latent(&mut rng)).collect())
        .collect();
    let slots = SyntheticSlots::build_based(dev, &windows, &vec![base; cfg.slots])?;
    let hdr_ptr = slots.header_device_ptr()?;

    // Incoming token per slot (bf16 → FP8 scatter inside the kernel) + queries.
    let kv_raw: Vec<[f32; HEAD_DIM]> = (0..cfg.slots).map(|_| bf16_latent(&mut rng)).collect();
    let kv_new = latents_2d(dev, &kv_raw)?;
    let q_raw: Vec<[f32; HEAD_DIM]> = (0..cfg.slots * N_HEADS)
        .map(|_| bf16_latent(&mut rng))
        .collect();
    let q = latents_3d(dev, &q_raw, cfg.slots, N_HEADS)?;

    // Compressed corpus at depth D + per-slot scattered selection. The decode
    // assembly mirrors the wave: each slot's `topk` gather concatenated, with
    // `comp_idx` pointing at each slot's block (offset `i·topk`). One gather of
    // all `slots·topk` gids (in slot order) keeps setup off the CPU-spill hot
    // path — the loop we actually profile is the kernel, not the gather.
    let g_total = (cfg.depth_tokens / cfg.ratio).max(cfg.topk);
    let gallery = build_gallery(dev, &mut rng, g_total, cfg.ratio)?;
    let all_gids: Vec<u32> = (0..cfg.slots)
        .flat_map(|_| scattered_gids(&mut rng, g_total, cfg.topk))
        .collect();
    let gids_t = Tensor::from_vec(all_gids, cfg.slots * cfg.topk, dev)?;
    let (comp, comp_pos) = gallery.gather_selected(&gids_t)?; // [slots*topk, 512], [slots*topk]
    let idx: Vec<u32> = (0..(cfg.slots * cfg.topk) as u32).collect();
    let comp_idx = Tensor::from_vec(idx, (cfg.slots, cfg.topk), dev)?;
    let comp_cnt = Tensor::from_vec(vec![cfg.topk as u32; cfg.slots], cfg.slots, dev)?;

    let sinks = Tensor::from_vec(
        (0..N_HEADS).map(|_| rng.sym() * 0.5).collect::<Vec<_>>(),
        N_HEADS,
        dev,
    )?;
    let rope_tab = csa_rope_table(dev)?;
    let ws = LatentWorkspace::build(dev)?;
    eprintln!(
        "[bench] decode setup done in {:.2}s",
        t_setup.elapsed().as_secs_f64()
    );

    // The persistent int8 corpus cache the kernel reads: gather the gallery's
    // pre-built two-region directly (the production path — tier-aware, no
    // rebuild). The f32 `comp` above is retained only for the host reference gate.
    let (ni8, nsc, rbf, cpos) = gallery.gather_corpus(&gids_t)?;
    let cache = CorpusCache::from_gathered(ni8, nsc, rbf, cpos, cfg.slots * cfg.topk)?;
    // Query position per slot = the writer-slice position the kernel used to
    // derive (window based at `base`, WINDOW tokens → `base + WINDOW`).
    let q_pos_t = Tensor::from_vec(vec![(base + WINDOW) as u32; cfg.slots], cfg.slots, dev)?;
    let launch = || -> Result<Tensor> {
        paged_latent_decode_raw(
            &q, hdr_ptr, &kv_new, &cache, &comp_idx, &comp_cnt, &q_pos_t, &sinks, &rope_tab,
            scale, WINDOW, cfg.splits, false, &ws, None,
        )
    };

    for _ in 0..cfg.warmup {
        launch()?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..cfg.iters {
        last = Some(launch()?);
    }
    dev.synchronize()?;
    let elapsed = t0.elapsed().as_secs_f64();

    let out = last.unwrap();
    let t_gate = Instant::now();
    let max_rel_err = gate_decode(
        &out, &windows, &kv_raw, &q_raw, &comp, &comp_pos, &sinks, &rope_tab, base, cfg, scale,
    )?;
    eprintln!(
        "[bench] decode gate done in {:.2}s",
        t_gate.elapsed().as_secs_f64()
    );

    let keys_per_query = WINDOW + cfg.topk;
    let tokens_per_s = (cfg.slots * cfg.iters) as f64 / elapsed;
    Ok(Report {
        label: format!(
            "decode  slots={} depth={}k topk={}",
            cfg.slots,
            cfg.depth_tokens / 1000,
            cfg.topk
        ),
        launches: cfg.iters,
        keys_per_query,
        gallery_entries: g_total,
        spilled: gallery.is_spilled(),
        per_call_ms: elapsed / cfg.iters as f64 * 1e3,
        tokens_per_s,
        keys_per_s: tokens_per_s * keys_per_query as f64,
        max_rel_err,
    })
}

#[allow(clippy::too_many_arguments)]
fn gate_decode(
    out: &Tensor,
    windows: &[Vec<[f32; HEAD_DIM]>],
    kv_raw: &[[f32; HEAD_DIM]],
    q_raw: &[[f32; HEAD_DIM]],
    comp: &Tensor,
    comp_pos: &Tensor,
    sinks: &Tensor,
    rope_tab: &Tensor,
    base: usize,
    cfg: DecodeCfg,
    scale: f32,
) -> Result<f32> {
    let tab = rope_tab
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let sinks_v = sinks.to_vec1::<f32>()?;
    let out = out.to_dtype(DType::F32)?; // [slots, H, 512]
    let q_pos = base + WINDOW;
    let mut worst = 0.0f32;

    let spot_slots: Vec<usize> = [0, cfg.slots / 2, cfg.slots - 1].into_iter().collect();
    let spot_heads = [0usize, 1, 31, 63];
    for &s in spot_slots.iter().filter(|&&s| s < cfg.slots) {
        // Keys are head-independent (MQA single latent): build once per slot.
        let comp_s = comp.narrow(0, s * cfg.topk, cfg.topk)?.to_vec2::<f32>()?;
        let pos_s = comp_pos
            .narrow(0, s * cfg.topk, cfg.topk)?
            .to_vec1::<u32>()?;
        let mut keys: Vec<[f32; HEAD_DIM]> = Vec::with_capacity(WINDOW + cfg.topk);
        // Window t=1..127 (t=0 falls just outside the clamp) + incoming.
        for t in 1..WINDOW {
            let mut k = windows[s][t];
            rope_fwd(&mut k, base + t, &tab);
            keys.push(k);
        }
        let mut inc: [f32; HEAD_DIM] =
            std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(kv_raw[s][d])));
        rope_fwd(&mut inc, base + WINDOW, &tab);
        keys.push(inc);
        for (e, row) in comp_s.iter().enumerate() {
            let mut k: [f32; HEAD_DIM] = std::array::from_fn(|d| row[d]);
            rope_fwd(&mut k, pos_s[e] as usize, &tab);
            keys.push(k);
        }
        for &h in spot_heads.iter().filter(|&&h| h < N_HEADS) {
            let mut qv = q_raw[s * N_HEADS + h];
            rope_fwd(&mut qv, q_pos, &tab);
            let r = ref_head(&qv, &keys, sinks_v[h], scale, q_pos, &tab);
            let got = out
                .narrow(0, s, 1)?
                .narrow(1, h, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            worst = worst.max(rel_err(&got, &r));
        }
    }
    if worst >= GATE_TOL {
        candle::bail!("decode correctness gate FAILED: max_rel_err {worst} ≥ {GATE_TOL}");
    }
    Ok(worst)
}

// ─── prefill harness ──────────────────────────────────────────────────────────

pub fn run_prefill(dev: &Device, cfg: PrefillCfg) -> Result<Report> {
    if cfg.topk > 512 {
        candle::bail!("prefill topk {} exceeds index_topk 512", cfg.topk);
    }
    let t_setup = Instant::now();
    let mut rng = Rng::new(cfg.seed);
    let base = cfg.depth_tokens.saturating_sub(WINDOW);
    let fresh_base = base + WINDOW; // new prompt starts at depth D
    let scale = (HEAD_DIM as f64).powf(-0.5) as f32;

    // Settled window (the existing deep context's last 128 tokens) in ONE slot.
    let window: Vec<[f32; HEAD_DIM]> = (0..WINDOW).map(|_| fp8_latent(&mut rng)).collect();
    let slots = SyntheticSlots::build_based(dev, std::slice::from_ref(&window), &[base])?;

    // New prompt: `total_q` fresh latents (bf16, FP8-round-tripped in-kernel) +
    // their queries, at positions `[D, D+total_q)`.
    let fresh_raw: Vec<[f32; HEAD_DIM]> = (0..cfg.total_q).map(|_| bf16_latent(&mut rng)).collect();
    let kv_fresh = latents_2d(dev, &fresh_raw)?;
    let q_raw: Vec<[f32; HEAD_DIM]> = (0..cfg.total_q * N_HEADS)
        .map(|_| bf16_latent(&mut rng))
        .collect();
    let q = latents_3d(dev, &q_raw, cfg.total_q, N_HEADS)?;
    let q_pos = Tensor::from_vec(
        (fresh_base as u32..(fresh_base + cfg.total_q) as u32).collect::<Vec<_>>(),
        cfg.total_q,
        dev,
    )?;

    // Deep compressed corpus, gathered whole (union) as the wave does; each
    // query points `comp_idx` at a scattered top-`topk` subset.
    let g_total = (cfg.depth_tokens / cfg.ratio).max(cfg.topk);
    let gallery = build_gallery(dev, &mut rng, g_total, cfg.ratio)?;
    let all = Tensor::arange(0u32, g_total as u32, dev)?;
    let (comp, comp_pos) = gallery.gather_selected(&all)?; // [g_total, 512], [g_total]
    let mut gids_per_q: Vec<Vec<u32>> = Vec::with_capacity(cfg.total_q);
    let mut idx: Vec<u32> = Vec::with_capacity(cfg.total_q * cfg.topk);
    for _ in 0..cfg.total_q {
        let gids = scattered_gids(&mut rng, g_total, cfg.topk);
        idx.extend_from_slice(&gids);
        gids_per_q.push(gids);
    }
    let comp_idx = Tensor::from_vec(idx, (cfg.total_q, cfg.topk), dev)?;
    let comp_cnt = Tensor::from_vec(vec![cfg.topk as u32; cfg.total_q], cfg.total_q, dev)?;

    let sinks = Tensor::from_vec(
        (0..N_HEADS).map(|_| rng.sym() * 0.5).collect::<Vec<_>>(),
        N_HEADS,
        dev,
    )?;
    let rope_tab = csa_rope_table(dev)?;
    let ws = LatentWorkspace::build(dev)?;
    eprintln!(
        "[bench] prefill setup done in {:.2}s",
        t_setup.elapsed().as_secs_f64()
    );

    // The kernel cache: gather the gallery's pre-built two-region directly (the
    // production path — tier-aware, no rebuild). The f32 `comp` above is retained
    // only for the host reference gate.
    let (ni8, nsc, rbf, cpos) = gallery.gather_corpus(&all)?;
    let cache = CorpusCache::from_gathered(ni8, nsc, rbf, cpos, g_total)?;
    let launch = || -> Result<Tensor> {
        paged_latent_prefill(
            &q,
            &slots.headers,
            &q_pos,
            Some((&kv_fresh, fresh_base)),
            &cache,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &rope_tab,
            &ws,
            scale,
            WINDOW,
            0,
            fp8_store_tag(),
        )
    };

    for _ in 0..cfg.warmup {
        launch()?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..cfg.iters {
        last = Some(launch()?);
    }
    dev.synchronize()?;
    let elapsed = t0.elapsed().as_secs_f64();

    let out = last.unwrap();
    let t_gate = Instant::now();
    let max_rel_err = gate_prefill(
        &out,
        &window,
        &fresh_raw,
        &q_raw,
        &comp,
        &comp_pos,
        &gids_per_q,
        &sinks,
        &rope_tab,
        base,
        cfg,
        scale,
    )?;
    eprintln!(
        "[bench] prefill gate done in {:.2}s",
        t_gate.elapsed().as_secs_f64()
    );

    let keys_per_query = WINDOW + cfg.topk;
    let tokens_per_s = (cfg.total_q * cfg.iters) as f64 / elapsed;
    Ok(Report {
        label: format!(
            "prefill total_q={} depth={}k topk={}",
            cfg.total_q,
            cfg.depth_tokens / 1000,
            cfg.topk
        ),
        launches: cfg.iters,
        keys_per_query,
        gallery_entries: g_total,
        spilled: gallery.is_spilled(),
        per_call_ms: elapsed / cfg.iters as f64 * 1e3,
        tokens_per_s,
        keys_per_s: tokens_per_s * keys_per_query as f64,
        max_rel_err,
    })
}

#[allow(clippy::too_many_arguments)]
fn gate_prefill(
    out: &Tensor,
    window: &[[f32; HEAD_DIM]],
    fresh_raw: &[[f32; HEAD_DIM]],
    q_raw: &[[f32; HEAD_DIM]],
    comp: &Tensor,
    comp_pos: &Tensor,
    gids_per_q: &[Vec<u32>],
    sinks: &Tensor,
    rope_tab: &Tensor,
    base: usize,
    cfg: PrefillCfg,
    scale: f32,
) -> Result<f32> {
    let tab = rope_tab
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let sinks_v = sinks.to_vec1::<f32>()?;
    let comp_pos_v = comp_pos.to_vec1::<u32>()?;
    let out = out.to_dtype(DType::F32)?; // [total_q, H, 512]
    let fresh_base = base + WINDOW;
    let mut worst = 0.0f32;

    let spot_q: Vec<usize> = [0, cfg.total_q / 2, cfg.total_q - 1].into_iter().collect();
    let spot_heads = [0usize, 1, 31, 63];
    for &qi in spot_q.iter().filter(|&&q| q < cfg.total_q) {
        let pi = fresh_base + qi;
        let clamp_lo = pi.saturating_sub(WINDOW); // key pos must be > clamp_lo, ≤ pi
                                                  // Settled window keys still inside the slide.
        let mut keys: Vec<[f32; HEAD_DIM]> = Vec::new();
        for (t, w) in window.iter().enumerate() {
            let p = base + t;
            if p > clamp_lo && p <= pi {
                let mut k = *w;
                rope_fwd(&mut k, p, &tab);
                keys.push(k);
            }
        }
        // Fresh predecessors (causal + window), FP8-round-tripped like the kernel.
        for (j, fr) in fresh_raw.iter().enumerate().take(qi + 1) {
            let p = fresh_base + j;
            if p > clamp_lo && p <= pi {
                let mut k: [f32; HEAD_DIM] =
                    std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(fr[d])));
                rope_fwd(&mut k, p, &tab);
                keys.push(k);
            }
        }
        // Compressed (all selected, unconditional — group starts precede D).
        let gids = &gids_per_q[qi];
        let gids_t = Tensor::from_vec(gids.clone(), gids.len(), out.device())?;
        let sel = comp.index_select(&gids_t, 0)?.to_vec2::<f32>()?;
        for (e, row) in sel.iter().enumerate() {
            let mut k: [f32; HEAD_DIM] = std::array::from_fn(|d| row[d]);
            rope_fwd(&mut k, comp_pos_v[gids[e] as usize] as usize, &tab);
            keys.push(k);
        }
        for &h in spot_heads.iter().filter(|&&h| h < N_HEADS) {
            let mut qv = q_raw[qi * N_HEADS + h];
            rope_fwd(&mut qv, pi, &tab);
            let r = ref_head(&qv, &keys, sinks_v[h], scale, pi, &tab);
            let got = out
                .narrow(0, qi, 1)?
                .narrow(1, h, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            worst = worst.max(rel_err(&got, &r));
        }
    }
    if worst >= GATE_TOL {
        candle::bail!("prefill correctness gate FAILED: max_rel_err {worst} ≥ {GATE_TOL}");
    }
    Ok(worst)
}

/// max|a−b| normalized by the reference magnitude (scale-free residual).
fn rel_err(got: &[f32], reference: &[f32]) -> f32 {
    let scale = reference
        .iter()
        .fold(0.0f32, |m, &v| m.max(v.abs()))
        .max(1e-6);
    got.iter()
        .zip(reference)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()))
        / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seconds-scale smoke test: the decode harness runs and its output clears
    /// the correctness gate at a realistic (but small-iter) shape. Guards the
    /// harness against bit-rot as the kernels are optimized.
    #[test]
    #[ignore]
    fn decode_harness_smoke() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let cfg = DecodeCfg {
            slots: 16,
            depth_tokens: 200_000,
            topk: 512,
            ratio: 4,
            splits: 0,
            warmup: 3,
            iters: 10,
            seed: 1,
        };
        let r = run_decode(&dev, cfg)?;
        r.print();
        assert!(r.max_rel_err < GATE_TOL);
        Ok(())
    }

    #[test]
    #[ignore]
    fn prefill_harness_smoke() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let cfg = PrefillCfg {
            total_q: 1024,
            depth_tokens: 200_000,
            topk: 512,
            ratio: 4,
            warmup: 2,
            iters: 5,
            seed: 2,
        };
        let r = run_prefill(&dev, cfg)?;
        r.print();
        assert!(r.max_rel_err < GATE_TOL);
        Ok(())
    }
}
