//! Individual-kernel benchmark harness for the two-stage corpus **selection**
//! path (the decode-hot `two_stage_select`), the counterpart to [`super::bench`]
//! (which benches the attention kernel).
//!
//! Decode fires `two_stage_select` **once per session** per CSA layer — the
//! attention kernel is already batched to one launch, but the selection is not
//! (`kernel_attn_decode_prepare` → `two_stage_select`, per slot, in the wave
//! loop). Each session owns its gallery and, at decode time, every entry is
//! causal, so the per-session selections are independent and batchable with no
//! causal-mask divergence. This harness measures where the per-session launch
//! cost actually goes — timing each micro-kernel (`sign_pack`, `bdp_recall`,
//! `topm_select`) and each Stage-2 op (rescore matmul, argsort) **in isolation**
//! over a realistic 64-session batch — so a fused/batched replacement is
//! designed against measured cost, not a guess, and validated against the
//! per-session path it replaces (set-equal selected gids per session).
//!
//! Fast: it builds only the sign/key index each session's selection reads, at a
//! realistic depth, WITHOUT the 164 GB model — a run completes in seconds.

#![cfg(feature = "cuda")]

use std::time::Instant;

use candle::{DType, Device, Result, Tensor};

use super::gallery::{
    bdp_recall, gather_corpus_batched, sign_pack, topm_select, two_stage_select_batched,
    FloatGallery,
};
use super::kernel_attention::shortlist_m;
use super::paged::{NOPE_BANDS, NOPE_DIM, ROPE_DIM};

/// Indexer scoring-key width (config default).
const INDEX_HEAD_DIM: usize = 128;
/// Indexer scoring heads (config default).
const INDEX_N_HEADS: usize = 64;
/// Attended-entry width — the gallery requires `NOPE_DIM+ROPE_DIM = 512`.
const HEAD_DIM: usize = 512;
/// Histogram bins for `topm_select` (`n_heads·dim + 1`, the agreement range).
const BINS: usize = INDEX_N_HEADS * INDEX_HEAD_DIM + 1;

// ─── deterministic PRNG (splitmix64) ──────────────────────────────────────────

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
}

// ─── config / report ───────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct SelectCfg {
    /// Concurrent decode sessions in one wave (production target: 64).
    pub sessions: usize,
    /// Compressed gallery entries per session. Real decode has sessions at
    /// varied depths; the harness uses a common depth per run and also sweeps
    /// small→large so the launch-overhead vs compute crossover is visible.
    pub entries: usize,
    /// Indexer top-k (config default 512).
    pub top_k: usize,
    pub warmup: usize,
    pub iters: usize,
    pub seed: u64,
}

impl Default for SelectCfg {
    fn default() -> Self {
        Self {
            sessions: 64,
            entries: 8192,
            top_k: 512,
            warmup: 10,
            iters: 100,
            seed: 0x5E1E_C7ED,
        }
    }
}

/// Per-stage isolated timing over the whole session batch (one "call" = the
/// per-session loop the wave runs today: `sessions` launches of that stage).
pub struct SelectReport {
    pub label: String,
    pub sessions: usize,
    pub entries: usize,
    pub shortlist_m: usize,
    /// Wall-time per whole-batch iteration, per stage (ms).
    pub sign_pack_ms: f64,
    pub bdp_recall_ms: f64,
    pub topm_ms: f64,
    pub rescore_ms: f64,
    pub argsort_ms: f64,
    /// Full `two_stage_select` per-session loop (the sum, cross-check).
    pub full_loop_ms: f64,
    /// Batched path over the whole session batch (0 until wired).
    pub batched_ms: f64,
    /// Fraction of sessions whose batched selected-gid SET matches the
    /// per-session loop (1.0 = exact). 0 until the batched path is wired.
    pub match_frac: f32,
}

impl SelectReport {
    pub fn print(&self) {
        println!("── {} ──", self.label);
        println!(
            "  sessions={} entries/session={} shortlist_m={} top_k",
            self.sessions, self.entries, self.shortlist_m,
        );
        println!(
            "  per-session-loop stage timings (whole {}-session batch):",
            self.sessions
        );
        println!("    sign_pack   {:>8.3} ms", self.sign_pack_ms);
        println!("    bdp_recall  {:>8.3} ms", self.bdp_recall_ms);
        println!("    topm_select {:>8.3} ms", self.topm_ms);
        println!(
            "    rescore     {:>8.3} ms  (matmul+relu+mul+sum)",
            self.rescore_ms
        );
        println!(
            "    argsort×2   {:>8.3} ms  (+index_select)",
            self.argsort_ms
        );
        println!("    ── full loop {:>8.3} ms", self.full_loop_ms);
        if self.batched_ms > 0.0 {
            println!(
                "  BATCHED      {:>8.3} ms   ({:.2}× vs loop), match={:.1}%",
                self.batched_ms,
                self.full_loop_ms / self.batched_ms,
                self.match_frac * 100.0,
            );
        }
    }
}

// ─── synthetic per-session galleries ──────────────────────────────────────────

/// Build `sessions` galleries, each with `entries` compressed rows, plus the
/// per-session Indexer query `[h, ih]` + gate weights `[h]`.
///
/// The keys are **correlated with the query** the way the real Indexer's are —
/// sign agreement ↔ float score is exactly what makes BDP recall work. The
/// selection is only *well-posed* (a deterministic top-k, so batched≡loop is
/// meaningful) when the truly-relevant entries sit unambiguously inside the
/// recall shortlist rather than at its tie-riddled boundary: agreement has only
/// ~`ih` distinct values, so with `entries ≫ ih` boundary ties are unavoidable
/// and the loop *itself* is non-deterministic there (arbitrary tie-fill). So we
/// model a realistic **relevant group** of `R` entries with `top_k < R <
/// shortlist_m`: each session picks a sign target `t`; the relevant entries key
/// to `t·(1+aᵢ)` (agreement ≈ max, distinct scores via `aᵢ`), the rest to a weak
/// `t·ε + noise` (clearly lower agreement). Every valid top-M shortlist then
/// contains ALL of `R` (they are the unambiguous agreement leaders, and `R ≤
/// shortlist_m`), so both paths rescore the same relevant set and pick the same
/// top-k — while boundary ties fall only among irrelevant fillers that never
/// enter the top-k.
fn build_batch(
    dev: &Device,
    cfg: SelectCfg,
) -> Result<(Vec<FloatGallery>, Vec<Tensor>, Vec<Tensor>)> {
    let mut rng = Rng::new(cfg.seed);
    let m = shortlist_m(cfg.top_k).min(cfg.entries);
    // Relevant group: strictly between top_k and the shortlist width, so it is
    // fully captured by any valid recall shortlist with room to spare.
    let relevant = ((cfg.top_k + m) / 2)
        .min(cfg.entries)
        .max(cfg.top_k.min(cfg.entries));
    let mut galleries = Vec::with_capacity(cfg.sessions);
    let mut queries = Vec::with_capacity(cfg.sessions);
    let mut weights = Vec::with_capacity(cfg.sessions);
    for _ in 0..cfg.sessions {
        let t: Vec<f32> = (0..INDEX_HEAD_DIM)
            .map(|_| if rng.bits() & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        let qv: Vec<f32> = (0..INDEX_N_HEADS)
            .flat_map(|_| {
                t.iter()
                    .map(|&td| td * (rng.sym().abs() + 0.05))
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(qv, (INDEX_N_HEADS, INDEX_HEAD_DIM), dev)?;
        let wv: Vec<f32> = (0..INDEX_N_HEADS)
            .map(|_| rng.sym().abs() * 0.1 + 0.01)
            .collect();
        let w = Tensor::from_vec(wv, INDEX_N_HEADS, dev)?;

        let mut g = FloatGallery::new(dev, HEAD_DIM, INDEX_HEAD_DIM, 8192)?;
        let mut keys = Vec::with_capacity(cfg.entries * INDEX_HEAD_DIM);
        for i in 0..cfg.entries {
            if i < relevant {
                // Relevant: aligned with `t`, distinct increasing score.
                let a = 1.0 + (i as f32 + 1.0) / relevant as f32;
                for &td in &t {
                    keys.push(td * a + rng.sym() * 0.02);
                }
            } else {
                // Irrelevant: weak alignment, noise-dominated → lower agreement.
                for &td in &t {
                    keys.push(td * 0.02 + rng.sym());
                }
            }
        }
        let attn: Vec<f32> = (0..cfg.entries * HEAD_DIM).map(|_| rng.sym()).collect();
        let positions: Vec<u32> = (0..cfg.entries).map(|i| (i * 4) as u32).collect();
        let attn_t = Tensor::from_vec(attn, (cfg.entries, HEAD_DIM), dev)?;
        let keys_t = Tensor::from_vec(keys, (cfg.entries, INDEX_HEAD_DIM), dev)?;
        g.append_batch(&attn_t, &keys_t, &positions)?;

        galleries.push(g);
        queries.push(q);
        weights.push(w);
    }
    Ok((galleries, queries, weights))
}

/// Rigorous, deterministic kernel gate (independent of the arbitrary recall
/// tie-fill): the batched `bdp_recall` counts must be **bit-exact** vs the
/// per-session `bdp_recall`, and the batched `topm` shortlist must be a **valid
/// top-M** per session (its selected count-multiset equals the true top-M's).
/// Returns `(counts_bit_exact, topm_valid_frac)`.
pub fn validate_kernels(dev: &Device, cfg: SelectCfg) -> Result<(bool, f32)> {
    use super::gallery::{bdp_recall_batched, topm_select_batched};
    let (galleries, queries, _w) = build_batch(dev, cfg)?;
    let n = cfg.sessions;
    let max_m = shortlist_m(cfg.top_k).min(cfg.entries);

    // Concatenate the batched Stage-1 inputs.
    let q_signs = Tensor::cat(
        &queries.iter().map(sign_pack).collect::<Result<Vec<_>>>()?,
        0,
    )?;
    let g_signs: Vec<Tensor> = galleries
        .iter()
        .map(|g| g.packed_signs())
        .collect::<Result<_>>()?;
    let signs_cat = Tensor::cat(&g_signs, 0)?;
    let off: Vec<u32> = (0..n).map(|s| (s * cfg.entries) as u32).collect();
    let cnt: Vec<u32> = vec![cfg.entries as u32; n];
    let off_t = Tensor::from_vec(off, n, dev)?;
    let cnt_t = Tensor::from_vec(cnt, n, dev)?;

    let counts_b = bdp_recall_batched(
        &q_signs,
        &signs_cat,
        &off_t,
        &cnt_t,
        n,
        INDEX_N_HEADS,
        cfg.entries,
        INDEX_HEAD_DIM,
    )?
    .to_vec1::<u32>()?;

    // Per-session counts (the trusted reference) + validity of the batched topm.
    let short_b = topm_select_batched(
        &Tensor::from_vec(counts_b.clone(), counts_b.len(), dev)?,
        &off_t,
        &cnt_t,
        n,
        cfg.entries,
        max_m,
        BINS,
    )?;
    let mut counts_exact = true;
    let mut valid = 0usize;
    for (s, (g, q)) in galleries.iter().zip(&queries).enumerate() {
        let qs = sign_pack(q)?;
        let counts_ref = bdp_recall(&qs, &g.packed_signs()?, INDEX_HEAD_DIM)?.to_vec1::<u32>()?;
        let counts_seg = &counts_b[s * cfg.entries..(s + 1) * cfg.entries];
        if counts_seg != counts_ref.as_slice() {
            counts_exact = false;
        }
        // Valid top-M: selected entries' counts, sorted desc, equal the true
        // top-M counts, sorted desc (tie-agnostic — any M-superset is valid).
        let sel = short_b.narrow(0, s, 1)?.reshape(max_m)?.to_vec1::<u32>()?;
        let mut sel_counts: Vec<u32> = sel.iter().map(|&i| counts_ref[i as usize]).collect();
        sel_counts.sort_unstable_by(|a, b| b.cmp(a));
        let mut all = counts_ref.clone();
        all.sort_unstable_by(|a, b| b.cmp(a));
        if sel_counts[..] == all[..max_m] {
            valid += 1;
        }
    }
    Ok((counts_exact, valid as f32 / n as f32))
}

/// Time a per-batch closure: `warmup` then `iters` calls, returns ms/iter.
fn time_batch<F: FnMut() -> Result<()>>(dev: &Device, cfg: SelectCfg, mut f: F) -> Result<f64> {
    for _ in 0..cfg.warmup {
        f()?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..cfg.iters {
        f()?;
    }
    dev.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / cfg.iters as f64 * 1e3)
}

/// Selected gids (ascending) per session via the current per-session
/// `two_stage_select` — the reference the batched path must reproduce.
fn reference_gids(
    galleries: &[FloatGallery],
    queries: &[Tensor],
    weights: &[Tensor],
    top_k: usize,
) -> Result<Vec<Vec<u32>>> {
    let m = shortlist_m(top_k);
    galleries
        .iter()
        .zip(queries)
        .zip(weights)
        .map(|((g, q), w)| {
            let (gids, k) = g.two_stage_select(q, w, m, top_k)?;
            Ok(gids.narrow(0, 0, k)?.to_vec1::<u32>()?)
        })
        .collect()
}

// ─── the harness ──────────────────────────────────────────────────────────────

pub fn run_select(dev: &Device, cfg: SelectCfg) -> Result<SelectReport> {
    let t_setup = Instant::now();
    let (galleries, queries, weights) = build_batch(dev, cfg)?;
    let m = shortlist_m(cfg.top_k);
    eprintln!(
        "[select-bench] setup done in {:.2}s ({} sessions × {} entries)",
        t_setup.elapsed().as_secs_f64(),
        cfg.sessions,
        cfg.entries,
    );

    // Pre-pack each session's query signs + hold each gallery's packed-sign view
    // so the per-stage timers isolate exactly one kernel.
    let q_signs: Vec<Tensor> = queries.iter().map(sign_pack).collect::<Result<_>>()?;
    let g_signs: Vec<Tensor> = galleries
        .iter()
        .map(|g| g.packed_signs())
        .collect::<Result<_>>()?;

    // Stage: sign_pack (the query, per session).
    let sign_pack_ms = time_batch(dev, cfg, || {
        for q in &queries {
            let _ = sign_pack(q)?;
        }
        Ok(())
    })?;

    // Stage: bdp_recall (per session, full gallery scan).
    let bdp_recall_ms = time_batch(dev, cfg, || {
        for (qs, gs) in q_signs.iter().zip(&g_signs) {
            let _ = bdp_recall(qs, gs, INDEX_HEAD_DIM)?;
        }
        Ok(())
    })?;

    // Precompute counts for the topm timer (isolate the histogram-select).
    let counts: Vec<Tensor> = q_signs
        .iter()
        .zip(&g_signs)
        .map(|(qs, gs)| bdp_recall(qs, gs, INDEX_HEAD_DIM))
        .collect::<Result<_>>()?;
    let topm_ms = time_batch(dev, cfg, || {
        for c in &counts {
            let _ = topm_select(c, m.min(cfg.entries), BINS)?;
        }
        Ok(())
    })?;

    // Stage 2 rescore (matmul+relu+broadcast_mul+sum) + argsort, over the
    // shortlist. Reuse a fixed shortlist per session (the exact ids don't change
    // the op cost) so the timers isolate the rescore/argsort kernels.
    let mm = m.min(cfg.entries);
    let shortlists: Vec<Tensor> = (0..cfg.sessions)
        .map(|_| Tensor::arange(0u32, mm as u32, dev))
        .collect::<Result<_>>()?;
    let keys: Vec<Tensor> = galleries
        .iter()
        .zip(&shortlists)
        .map(|(g, sl)| g.scoring_keys()?.index_select(sl, 0))
        .collect::<Result<_>>()?;
    let rescore_ms = time_batch(dev, cfg, || {
        for ((q, keys), w) in queries.iter().zip(&keys).zip(&weights) {
            let scores = q.matmul(&keys.t()?.contiguous()?)?.relu()?;
            let _ = scores.broadcast_mul(&w.reshape(((), 1))?)?.sum(0)?;
        }
        Ok(())
    })?;
    // Argsort×2 + index_select (the tail of two_stage_select), per session.
    let weighted: Vec<Tensor> = queries
        .iter()
        .zip(&keys)
        .zip(&weights)
        .map(|((q, keys), w)| {
            let scores = q.matmul(&keys.t()?.contiguous()?)?.relu()?;
            scores.broadcast_mul(&w.reshape(((), 1))?)?.sum(0)
        })
        .collect::<Result<_>>()?;
    let argsort_ms = time_batch(dev, cfg, || {
        for (wv, sl) in weighted.iter().zip(&shortlists) {
            let order = wv.unsqueeze(0)?.arg_sort_last_dim(false)?.squeeze(0)?;
            let picked = order.narrow(0, 0, cfg.top_k.min(mm))?.contiguous()?;
            let gids = sl.index_select(&picked, 0)?;
            let asc = gids
                .to_dtype(DType::F32)?
                .unsqueeze(0)?
                .arg_sort_last_dim(true)?
                .squeeze(0)?
                .contiguous()?;
            let _ = gids.index_select(&asc, 0)?;
        }
        Ok(())
    })?;

    // Full per-session loop (the wave's decode-selection cost today).
    let full_loop_ms = time_batch(dev, cfg, || {
        for ((g, q), w) in galleries.iter().zip(&queries).zip(&weights) {
            let _ = g.two_stage_select(q, w, m, cfg.top_k)?;
        }
        Ok(())
    })?;

    // ── Batched path: one launch per Stage-1 kernel over the whole wave ──
    let gref: Vec<&FloatGallery> = galleries.iter().collect();
    let batched_ms = time_batch(dev, cfg, || {
        let _ = two_stage_select_batched(&gref, &queries, &weights, m, cfg.top_k)?;
        Ok(())
    })?;

    // Correctness gate: per-session selected-gid SET equals the loop.
    let reference = reference_gids(&galleries, &queries, &weights, cfg.top_k)?;
    let batched = two_stage_select_batched(&gref, &queries, &weights, m, cfg.top_k)?;
    let batched_gids: Vec<Vec<u32>> = batched
        .iter()
        .map(|(g, k)| Ok(g.narrow(0, 0, *k)?.to_vec1::<u32>()?))
        .collect::<Result<_>>()?;
    let match_frac = gid_set_match(&batched_gids, &reference);

    Ok(SelectReport {
        label: format!("select sessions={} entries={}", cfg.sessions, cfg.entries),
        sessions: cfg.sessions,
        entries: cfg.entries,
        shortlist_m: m,
        sign_pack_ms,
        bdp_recall_ms,
        topm_ms,
        rescore_ms,
        argsort_ms,
        full_loop_ms,
        batched_ms,
        match_frac,
    })
}

/// Fraction of sessions whose selected-gid SET matches the reference. Used by
/// the correctness gate once the batched path lands.
pub fn gid_set_match(a: &[Vec<u32>], b: &[Vec<u32>]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut ok = 0usize;
    for (x, y) in a.iter().zip(b) {
        let mut xs = x.clone();
        let mut ys = y.clone();
        xs.sort_unstable();
        ys.sort_unstable();
        if xs == ys {
            ok += 1;
        }
    }
    ok as f32 / a.len() as f32
}

/// ncu / individual-kernel target for the fused corpus gather
/// (`corpus_gather_rows_kernel`): build `sessions` galleries, then loop the
/// per-session `gather_corpus_into` (one kernel launch each) that assembles the
/// decode wave's selected corpus block. Reports the whole-batch wall time.
pub fn run_corpus_gather_kernels(dev: &Device, cfg: SelectCfg, iters: usize) -> Result<()> {
    let (galleries, _q, _w) = build_batch(dev, cfg)?;
    let n = cfg.sessions;
    let k = cfg.top_k.min(cfg.entries); // selected rows per session
    let total = n * k;
    // Scattered gids per session (strided across the gallery, like a real
    // top-k selection spread over the whole depth).
    let stride = (cfg.entries / k).max(1);
    let gids: Vec<Tensor> = (0..n)
        .map(|_| {
            let v: Vec<u32> = (0..k)
                .map(|j| ((j * stride) % cfg.entries) as u32)
                .collect();
            Tensor::from_vec(v, k, dev)
        })
        .collect::<Result<_>>()?;
    // Output block reused across iterations (the kernel overwrites its rows).
    let out_nope = Tensor::zeros((total, NOPE_DIM), DType::U8, dev)?;
    let out_scale = Tensor::zeros((total, NOPE_BANDS), DType::F32, dev)?;
    let out_rope = Tensor::zeros((total, ROPE_DIM), DType::BF16, dev)?;
    let out_pos = Tensor::zeros(total, DType::U32, dev)?;
    let gref: Vec<&FloatGallery> = galleries.iter().collect();
    let offs: Vec<u32> = (0..n).map(|i| (i * k) as u32).collect();
    let one = |_: usize| -> Result<()> {
        gather_corpus_batched(
            &gref, &gids, &offs, &out_nope, &out_scale, &out_rope, &out_pos,
        )
    };
    for i in 0..cfg.warmup {
        one(i)?;
    }
    dev.synchronize()?;
    let t = Instant::now();
    for i in 0..iters {
        one(i)?;
    }
    dev.synchronize()?;
    let ms = t.elapsed().as_secs_f64() / iters as f64 * 1e3;
    println!(
        "[corpus-gather] {} sess × {} rows (from {} entries): {:.4} ms/iter (1 batched launch)",
        n, k, cfg.entries, ms
    );
    Ok(())
}

/// ncu / individual-kernel target: build a deep batch, then loop ONLY the two
/// batched Stage-1 selection kernels (`bdp_recall_batched` → `topm_select_batched`)
/// in isolation — no Stage-2, no per-session loop — so a profiler can attach to
/// `bdp_recall_batched_kernel` and the `topm_*_batched_kernel`s and nothing else.
/// Also reports the isolated `bdp+topm` wall time per iteration.
pub fn run_select_kernels(dev: &Device, cfg: SelectCfg, iters: usize) -> Result<()> {
    use super::gallery::{bdp_recall_batched, topm_select_batched};
    let (galleries, queries, _w) = build_batch(dev, cfg)?;
    let n = cfg.sessions;
    let max_m = shortlist_m(cfg.top_k).min(cfg.entries);
    let q_signs = Tensor::cat(
        &queries.iter().map(sign_pack).collect::<Result<Vec<_>>>()?,
        0,
    )?;
    let g_signs: Vec<Tensor> = galleries
        .iter()
        .map(|g| g.packed_signs())
        .collect::<Result<_>>()?;
    let signs = Tensor::cat(&g_signs, 0)?;
    let off = Tensor::from_vec(
        (0..n).map(|s| (s * cfg.entries) as u32).collect::<Vec<_>>(),
        n,
        dev,
    )?;
    let cnt = Tensor::from_vec(vec![cfg.entries as u32; n], n, dev)?;
    let one = |_: usize| -> Result<()> {
        let counts = bdp_recall_batched(
            &q_signs,
            &signs,
            &off,
            &cnt,
            n,
            INDEX_N_HEADS,
            cfg.entries,
            INDEX_HEAD_DIM,
        )?;
        let _ = topm_select_batched(&counts, &off, &cnt, n, cfg.entries, max_m, BINS)?;
        Ok(())
    };
    for i in 0..cfg.warmup {
        one(i)?;
    }
    dev.synchronize()?;
    let t = Instant::now();
    for i in 0..iters {
        one(i)?;
    }
    dev.synchronize()?;
    let ms = t.elapsed().as_secs_f64() / iters as f64 * 1e3;
    println!(
        "[select-kernels] {} sess × {} entries × {} heads: {:.4} ms/iter (bdp_recall_batched + topm_select_batched)",
        n, cfg.entries, INDEX_N_HEADS, ms
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seconds-scale smoke: the selection harness runs and its per-session
    /// reference is self-consistent (two_stage_select ⊇ nothing spurious). Prints
    /// the per-kernel table so the launch-cost breakdown is visible before any
    /// fusion — the "study the micro kernels they replace" step.
    #[test]
    #[ignore]
    fn select_harness_smoke() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for entries in [256usize, 2048, 8192] {
            let cfg = SelectCfg {
                sessions: 64,
                entries,
                top_k: 512,
                warmup: 5,
                iters: 30,
                seed: 7,
            };
            let r = run_select(&dev, cfg)?;
            r.print();
            // Rigorous, tie-fill-independent kernel gate.
            let (counts_exact, topm_valid) = validate_kernels(&dev, cfg)?;
            assert!(
                counts_exact,
                "batched bdp_recall counts diverge at entries={entries}"
            );
            assert!(
                topm_valid >= 0.999,
                "batched topm not a valid top-M at entries={entries}: {topm_valid:.3}"
            );
            // End-to-end: on correlated (realistic) keys the batched selection
            // reproduces the per-session loop's top-k exactly.
            assert!(
                r.match_frac >= 0.999,
                "batched select diverged from loop at entries={entries}: match={:.3}",
                r.match_frac
            );
            // Reference is well-formed: each session selects min(top_k, entries).
            let (g, q, w) = build_batch(&dev, cfg)?;
            let gids = reference_gids(&g, &q, &w, cfg.top_k)?;
            assert_eq!(gids.len(), cfg.sessions);
            for row in &gids {
                assert_eq!(row.len(), cfg.top_k.min(cfg.entries));
            }
        }
        Ok(())
    }
}
