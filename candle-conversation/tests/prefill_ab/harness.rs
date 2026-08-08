//! Shared harness for the prefill kernel tests.
//!
//! Builds paged-KV prefill problems whose prefix is **genuine palette-
//! quantized arena content** (produced by the real seal + `quantize_sealed_
//! in_place` path — the same bytes production attends over), runs a prefill
//! kernel backend over them, and compares against the CPU golden reference
//! (`golden.rs`) and between backends.
//!
//! No model is loaded anywhere: K/V/Q values are seeded synthetic data (or,
//! via `substrate_source.rs`, real sealed chunks recovered from a substrate
//! redo log), so the harness isolates the *kernel* — arena addressing,
//! palette routing, dequantization, RoPE, and the attention math itself.

use candle::quantized::pinned_staging::{PinnedBuf, PinnedStager};
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    quantize_sealed_in_place, ChunkedKvBacking, CompressionPolicy, KvCache, CHUNK_SIZE,
};
use candle_transformers::models::prefill_utils::{compute_rope_cs, paged_prefill_batched};
use std::sync::{Mutex, MutexGuard};

// Production attention shape (Qwen3-MoE): GQA 32/4, HEAD_DIM 128. The GQA
// ratio matters — the new kernel packs the group into the MMA M dimension,
// so the harness must exercise hpg > 1 from day one.
pub const N_HEAD: usize = 32;
pub const N_KV_HEAD: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const MAX_BLOCKS: usize = 512;

/// Tests share one GPU and the process-global quantized arena table; they
/// must not run concurrently (same pattern as `kernel_layout_tests`).
static GPU_SERIAL: Mutex<()> = Mutex::new(());

pub fn gpu_serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

// ──────────────────────────────────────────────────────────────────────
// Scenario description
// ──────────────────────────────────────────────────────────────────────

/// One sealed prefix segment: `len` tokens quantized at compression level
/// `level` (`None` = leave as F16 arena chunks — the identity path that
/// validates the golden reference itself, RoPE convention included).
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub len: usize,
    pub level: Option<u8>,
}

/// One sequence in the batch: its sealed-prefix layout plus the fresh
/// prefill (`q_len` new tokens attending over prefix + themselves).
#[derive(Clone, Debug)]
pub struct SeqSpec {
    pub segments: Vec<Segment>,
    pub q_len: usize,
}

impl SeqSpec {
    pub fn prefix_len(&self) -> usize {
        self.segments.iter().map(|s| s.len).sum()
    }
}

#[derive(Clone, Debug)]
pub struct Scenario {
    pub name: &'static str,
    pub seqs: Vec<SeqSpec>,
    /// RoPE theta. `0.0` = identity RoPE (inv_freq = 0 ⇒ cos 1 / sin 0),
    /// isolating arena/dequant behavior from rotation; otherwise the
    /// standard `1 / theta^(2i/d)` table (1e6 = production-like).
    pub theta: f64,
    pub seed: u64,
    /// Acceptance band for `golden` comparison, as max |gpu − golden|
    /// normalized by the golden's max magnitude. Sized per scenario by the
    /// quantization level of its prefix (an F16 identity prefix is tight;
    /// a C7 prefix carries the quantizer's full error budget).
    ///
    /// NOTE on sizing: the synthetic K/V is uniform random — the
    /// quantizer's adversarial case (no channel structure for the palette
    /// selection to exploit), so these bands are deliberately looser than
    /// production error. The golden leg is a sanity band; bitwise
    /// reset-rerun determinism over identical arena bytes is the exact
    /// oracle.
    pub golden_band: f32,
    /// Minimum acceptable per-row cosine vs golden — per-scenario for the
    /// same reason as `golden_band`.
    pub min_cos: f32,
    /// Apply deterministic magnitude structure to the K (per-dim) and V
    /// (per-token) sources before sealing. Uniform-random values make the
    /// compression policy pick ONE format for every dim band — trivial
    /// palette routing. The profiles interleave magnitude classes at
    /// stride 1, so per-band format choices differ and the palette maps
    /// SCATTER (dim → palette assignments interleave instead of forming
    /// contiguous bands) — the adversarial case for the kernel's rank
    /// tables, produced through the production seal path.
    pub structured_dims: bool,
}

/// Per-dim K magnitude profile (K is channel-sensitive): four magnitude
/// classes interleaved at stride 1 in d, with a slow per-dim wobble so no
/// two dims of a class are identical. Re-rounds through F16 to preserve
/// the host/arena exactness invariant of `Rng::f16_unit`.
pub fn apply_k_profile(host: &mut [f32], n_tok: usize) {
    const S: [f32; 4] = [0.02, 1.0, 0.15, 6.0];
    for row in 0..n_tok * N_KV_HEAD {
        for d in 0..HEAD_DIM {
            let s = S[d % 4] * (1.0 + 0.25 * ((d / 4) as f32 * 0.7).sin());
            let x = &mut host[row * HEAD_DIM + d];
            *x = f32::from(half::f16::from_f32(*x * s));
        }
    }
}

/// Per-token V magnitude profile (V is token-sensitive), with a mild
/// per-dim alternation on top. `tok_base` is the sequence-global position
/// of the segment's first token so the profile does not repeat per
/// segment.
pub fn apply_v_profile(host: &mut [f32], n_tok: usize, tok_base: usize) {
    const SD: [f32; 2] = [1.0, 0.08];
    const ST: [f32; 3] = [1.0, 0.05, 3.0];
    for t in 0..n_tok {
        let st = ST[(tok_base + t) % 3] * (1.0 + 0.2 * ((tok_base + t) as f32 * 0.31).cos());
        for h in 0..N_KV_HEAD {
            for d in 0..HEAD_DIM {
                let x = &mut host[(t * N_KV_HEAD + h) * HEAD_DIM + d];
                *x = f32::from(half::f16::from_f32(*x * st * SD[d % 2]));
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Deterministic value generation (no model anywhere)
// ──────────────────────────────────────────────────────────────────────

/// xorshift64* — deterministic across platforms, seeded per scenario.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [-1, 1), rounded through F16 so the host-side golden and
    /// the F16 arena/Q tensors see the *same* values.
    pub fn f16_unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        f32::from(half::f16::from_f32(u * 2.0 - 1.0))
    }
    pub fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f16_unit()).collect()
    }
    /// Uniform integer in [0, n) — for the seeded fuzz sweeps.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ──────────────────────────────────────────────────────────────────────
// Built case
// ──────────────────────────────────────────────────────────────────────

/// A fully-built prefill problem: populated arena + host-side source values
/// for the golden. Rebuilt fresh per kernel run — a prefill *writes* its new
/// tokens into the arena, so a built case is single-shot by design.
pub struct BuiltCase {
    pub backing: ChunkedKvBacking,
    pub caches: Vec<KvCache>,
    /// Sealed prefix chunk count per slot — the truncation point
    /// `reset_to_prefix` restores to. A prefill writes its new tokens into
    /// fresh chunks appended after the sealed prefix (sealed chunks are
    /// immutable; a partial sealed tail is a gap, never a write target), so
    /// truncating back to this count restores the arena byte-for-byte.
    pub prefix_chunks: Vec<usize>,
    /// Host source values per sequence, all pre-rotation (K stored
    /// position-independent, exactly as the arena stores it):
    /// prefix K/V `[prefix][N_KV_HEAD][HEAD_DIM]`,
    /// new Q `[q_len][N_HEAD][HEAD_DIM]`, new K/V `[q_len][N_KV_HEAD][HEAD_DIM]`.
    pub prefix_k: Vec<Vec<f32>>,
    pub prefix_v: Vec<Vec<f32>>,
    pub new_q: Vec<Vec<f32>>,
    pub new_k: Vec<Vec<f32>>,
    pub new_v: Vec<Vec<f32>>,
    pub rope_cs_host: Vec<f32>,
    pub rope_cs: Tensor,
    pub rope_offsets: Tensor,
    /// Pinned staging pool, built once per case — `PinnedStager::new` is a
    /// pinned host allocation and must stay out of the per-run timing.
    pub stager: PinnedStager,
    /// Device-resident packed inputs + varlen metadata, built once — the
    /// production shape: Q/K/V arrive from on-device projections and the
    /// scheduler prebuilds prefill_meta, so neither belongs in a timed run.
    pub q_dev: Tensor,
    pub k_dev: Tensor,
    pub v_dev: Tensor,
    pub cu_seqlens_q: Tensor,
    pub q_lens_dev: Tensor,
    pub kv_lens_dev: Tensor,
    pub spec: Scenario,
}

fn host_to_bhtd(host: &[f32], n_heads: usize, n_tok: usize, device: &Device) -> Result<Tensor> {
    // host layout is [tok][head][dim] → tensor [1, n_heads, n_tok, HEAD_DIM]
    let t = Tensor::from_vec(host.to_vec(), (n_tok, n_heads, HEAD_DIM), device)?;
    t.transpose(0, 1)?.unsqueeze(0)?.to_dtype(DType::F16)
}

/// Flatten `[tok][head][dim]` host values into the `[total, heads, dim]`
/// F16 layout the ragged prefill entry wants.
pub fn host_to_flat(host: &[f32], n_heads: usize, n_tok: usize, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(host.to_vec(), (n_tok, n_heads, HEAD_DIM), device)?.to_dtype(DType::F16)
}

fn cuda_stream(
    device: &Device,
) -> Result<std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>> {
    match device {
        Device::Cuda(d) => Ok(d.cuda_stream()),
        _ => candle::bail!("prefill_ab requires a CUDA device"),
    }
}

pub fn bind_cache(backing: &ChunkedKvBacking, batch_idx: usize) -> Result<KvCache> {
    let mut cache = KvCache::new(2, 64);
    cache.force_dtype(DType::F16);
    cache.set_chunked_backing(backing, batch_idx, None)?;
    Ok(cache)
}

/// Seal `len` tokens of K/V into `target_slot`'s tail as one segment,
/// optionally quantized at `level` through the production policy path.
/// Uses `scratch_slot` for the staging write (same pattern as
/// `kernel_layout_tests::build_segmented_slot` / the glue tests'
/// `build_sealed_arena_quant`).
#[allow(clippy::too_many_arguments)]
fn seal_segment(
    backing: &ChunkedKvBacking,
    target_slot: usize,
    scratch_slot: usize,
    k_host: &[f32],
    v_host: &[f32],
    len: usize,
    level: Option<u8>,
    device: &Device,
) -> Result<()> {
    let kr = host_to_bhtd(k_host, N_KV_HEAD, len, device)?.contiguous()?;
    let vr = host_to_bhtd(v_host, N_KV_HEAD, len, device)?.contiguous()?;
    backing.truncate_sequence_to_blocks(scratch_slot, 0)?;
    backing.ensure_for_offset(scratch_slot, 0, len)?;
    backing.write_contiguous(scratch_slot, 0, &kr, &vr)?;
    backing.set_len(scratch_slot, len);
    let real_chunks = len.div_ceil(CHUNK_SIZE).max(1);
    backing.truncate_sequence_to_blocks(scratch_slot, real_chunks)?;
    let sealed = backing.record_turn(scratch_slot)?;
    let sealed = match level {
        None => sealed,
        Some(l) => {
            let policy = CompressionPolicy::new(l);
            let copy_stream = cuda_stream(device)?;
            let mut scratch: Option<PinnedBuf> = None;
            let mut warm = quantize_sealed_in_place(
                backing,
                &[&sealed],
                &policy,
                device,
                &copy_stream,
                &mut scratch,
            )?;
            warm.remove(0)
        }
    };
    backing.truncate_sequence_to_blocks(scratch_slot, 0)?;
    backing.inject_sealed_at_tail(target_slot, &sealed)?;
    Ok(())
}

/// Build a scenario into a fresh backing: one slot per sequence with its
/// segmented, (optionally) quantized prefix sealed in, plus host-side source
/// values for the golden reference.
pub fn build_case(spec: &Scenario, device: &Device) -> Result<BuiltCase> {
    let n_seqs = spec.seqs.len();
    // slots: [0..n_seqs) = sequences, last = scratch for segment staging.
    let backing = ChunkedKvBacking::new(
        n_seqs + 1,
        N_KV_HEAD,
        HEAD_DIM,
        DType::F16,
        device,
        MAX_BLOCKS,
    )?;

    let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| {
            if spec.theta == 0.0 {
                0.0
            } else {
                (1.0 / spec.theta.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32
            }
        })
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, HEAD_DIM / 2, device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_cs_host = rope_cs
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let rope_offsets = Tensor::zeros(n_seqs, DType::U32, device)?;

    // Binding a cache to a slot is what allocates it — the scratch slot
    // needs one too before `seal_segment` can truncate/write it.
    let _scratch_cache = bind_cache(&backing, n_seqs)?;

    let mut caches = Vec::with_capacity(n_seqs);
    let mut prefix_chunks = Vec::with_capacity(n_seqs);
    let mut prefix_k = Vec::with_capacity(n_seqs);
    let mut prefix_v = Vec::with_capacity(n_seqs);
    let mut new_q = Vec::with_capacity(n_seqs);
    let mut new_k = Vec::with_capacity(n_seqs);
    let mut new_v = Vec::with_capacity(n_seqs);

    for (si, seq) in spec.seqs.iter().enumerate() {
        let mut rng = Rng::new(spec.seed ^ (0x9E3779B97F4A7C15u64.wrapping_mul(si as u64 + 1)));
        let mut cache = bind_cache(&backing, si)?;

        let mut pk_all: Vec<f32> = Vec::new();
        let mut pv_all: Vec<f32> = Vec::new();
        let mut sealed_total = 0usize;
        for seg in &seq.segments {
            let mut k_host = rng.fill(seg.len * N_KV_HEAD * HEAD_DIM);
            let mut v_host = rng.fill(seg.len * N_KV_HEAD * HEAD_DIM);
            if spec.structured_dims {
                apply_k_profile(&mut k_host, seg.len);
                apply_v_profile(&mut v_host, seg.len, sealed_total);
            }
            seal_segment(
                &backing, si, n_seqs, &k_host, &v_host, seg.len, seg.level, device,
            )?;
            sealed_total += seg.len;
            pk_all.extend_from_slice(&k_host);
            pv_all.extend_from_slice(&v_host);
        }
        cache.set_current_seq_len(sealed_total)?;
        debug_assert_eq!(sealed_total, seq.prefix_len());
        prefix_chunks.push(
            cache
                .k_cache()
                .chunked_live_chunks_as_sealed()
                .map(|c| c.len())
                .unwrap_or(0),
        );

        new_q.push(rng.fill(seq.q_len * N_HEAD * HEAD_DIM));
        let mut nk = rng.fill(seq.q_len * N_KV_HEAD * HEAD_DIM);
        let mut nv = rng.fill(seq.q_len * N_KV_HEAD * HEAD_DIM);
        if spec.structured_dims {
            apply_k_profile(&mut nk, seq.q_len);
            apply_v_profile(&mut nv, seq.q_len, sealed_total);
        }
        new_k.push(nk);
        new_v.push(nv);
        prefix_k.push(pk_all);
        prefix_v.push(pv_all);
        caches.push(cache);
    }

    let stager = match device {
        Device::Cuda(d) => PinnedStager::new(d),
        _ => candle::bail!("prefill_ab requires a CUDA device"),
    };

    // Device-resident packed inputs + varlen metadata (see the BuiltCase
    // field docs). Offsets are stable across runs — reset_to_prefix
    // restores every slot to the same sealed prefix.
    let (mut qs, mut ks, mut vs) = (Vec::new(), Vec::new(), Vec::new());
    for (si, seq) in spec.seqs.iter().enumerate() {
        qs.push(host_to_flat(&new_q[si], N_HEAD, seq.q_len, device)?);
        ks.push(host_to_flat(&new_k[si], N_KV_HEAD, seq.q_len, device)?);
        vs.push(host_to_flat(&new_v[si], N_KV_HEAD, seq.q_len, device)?);
    }
    let q_dev = Tensor::cat(&qs, 0)?.contiguous()?;
    let k_dev = Tensor::cat(&ks, 0)?.contiguous()?;
    let v_dev = Tensor::cat(&vs, 0)?.contiguous()?;
    let n_seqs = spec.seqs.len();
    let mut cu = Vec::with_capacity(n_seqs + 1);
    cu.push(0u32);
    let mut acc = 0u32;
    for seq in &spec.seqs {
        acc += seq.q_len as u32;
        cu.push(acc);
    }
    let cu_seqlens_q = Tensor::from_vec(cu, n_seqs + 1, device)?;
    let q_lens_dev = Tensor::from_vec(
        spec.seqs.iter().map(|s| s.q_len as u32).collect::<Vec<_>>(),
        n_seqs,
        device,
    )?;
    let kv_lens_dev = Tensor::from_vec(
        spec.seqs
            .iter()
            .map(|s| (s.prefix_len() + s.q_len) as u32)
            .collect::<Vec<_>>(),
        n_seqs,
        device,
    )?;
    Ok(BuiltCase {
        backing,
        caches,
        prefix_chunks,
        stager,
        prefix_k,
        prefix_v,
        new_q,
        new_k,
        new_v,
        rope_cs_host,
        rope_cs,
        rope_offsets,
        q_dev,
        k_dev,
        v_dev,
        cu_seqlens_q,
        q_lens_dev,
        kv_lens_dev,
        spec: spec.clone(),
    })
}

/// Rewind every slot to its sealed prefix, discarding the chunks a prior
/// kernel run appended for its new tokens. Sealed prefix chunks are
/// immutable (a partial sealed tail is a gap, never a write target), so
/// after this the arena is byte-identical to the freshly-built state — the
/// mechanism that lets every scenario rerun the kernel over the SAME
/// arena bytes for the bitwise determinism check.
pub fn reset_to_prefix(case: &mut BuiltCase) -> Result<()> {
    for (si, seq) in case.spec.seqs.iter().enumerate() {
        case.backing
            .truncate_sequence_to_blocks(si, case.prefix_chunks[si])?;
        case.caches[si].set_current_seq_len(seq.prefix_len())?;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Kernel backends
// ──────────────────────────────────────────────────────────────────────

/// Run one prefill over the built case through the production
/// `paged_prefill_batched` path (the INT8 prefix-attention kernel).
/// Returns the flat attention output `[total_q, N_HEAD, HEAD_DIM]` as F16.
pub fn run_prefill(case: &mut BuiltCase) -> Result<Tensor> {
    let n_seqs = case.spec.seqs.len();
    let generation = case.stager.begin_generation();

    // Device-resident inputs from the case (production shape — no H2D in
    // the run itself).
    let (q, k, v) = (case.q_dev.clone(), case.k_dev.clone(), case.v_dev.clone());
    let mut q_lens = Vec::with_capacity(n_seqs);
    let mut offsets = Vec::with_capacity(n_seqs);
    for (si, seq) in case.spec.seqs.iter().enumerate() {
        q_lens.push(seq.q_len);
        offsets.push(case.caches[si].current_seq_len());
    }

    let mut cache_refs: Vec<&mut KvCache> = case.caches.iter_mut().collect();
    let out = paged_prefill_batched(
        None,
        &mut cache_refs[..],
        &offsets,
        &q,
        &k,
        &v,
        n_seqs,
        &q_lens,
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        Some((&case.cu_seqlens_q, &case.q_lens_dev, &case.kv_lens_dev)),
        &case.rope_offsets,
        &case.rope_cs,
        false,
        &generation,
        &std::cell::RefCell::new(None),
    )?;
    for (si, seq) in case.spec.seqs.iter().enumerate() {
        let off = offsets[si];
        case.caches[si].set_current_seq_len(off + seq.q_len)?;
    }
    // F16 out, as production consumes it; comparison call sites convert.
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────
// Performance measurement
// ──────────────────────────────────────────────────────────────────────

/// Flatten a prefill output to F32 host values for comparison.
pub fn out_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// Time the prefill over a built case: `warmup` unmeasured runs, then
/// `reps` measured runs (device-synchronized), `reset_to_prefix` between
/// every run so each rep sees the identical arena. Returns (best, mean)
/// seconds per run — best-of is the comparison number (least noise), mean
/// the sanity check.
pub fn bench_prefill(
    case: &mut BuiltCase,
    device: &Device,
    warmup: usize,
    reps: usize,
) -> Result<(f64, f64)> {
    for _ in 0..warmup {
        reset_to_prefix(case)?;
        let _ = run_prefill(case)?;
        // Drain between warmups too: an unsynced warmup leaks its GPU tail
        // into the NEXT call's first profile_sync'd span, mis-attributing
        // kernel time to prefill:alloc in the profiled bench.
        device.synchronize()?;
    }
    let mut best = f64::INFINITY;
    let mut sum = 0f64;
    for _ in 0..reps {
        reset_to_prefix(case)?;
        device.synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = run_prefill(case)?;
        device.synchronize()?;
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        sum += dt;
    }
    Ok((best, sum / reps as f64))
}

// ──────────────────────────────────────────────────────────────────────
// Comparison metrics
// ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Metrics {
    /// max |a − b| over all elements, normalized by max |b|.
    pub max_rel: f32,
    /// minimum per-row (per q-token per head) cosine similarity.
    pub min_row_cos: f32,
}

pub fn compare(a: &[f32], b: &[f32]) -> Metrics {
    assert_eq!(a.len(), b.len(), "output length mismatch");
    let b_max = b.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-20);
    let max_abs = a
        .iter()
        .zip(b.iter())
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mut min_cos = f32::INFINITY;
    for (ra, rb) in a.chunks(HEAD_DIM).zip(b.chunks(HEAD_DIM)) {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-20);
        min_cos = min_cos.min((dot / denom) as f32);
    }
    Metrics {
        max_rel: max_abs / b_max,
        min_row_cos: min_cos,
    }
}
