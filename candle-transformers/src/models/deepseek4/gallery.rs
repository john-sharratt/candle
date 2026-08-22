//! Per-sequence compressed-corpus store + two-stage selection for
//! DeepSeek-V4-Flash: the **FloatGallery** holds the corpus pair — attended
//! entries `[G, head_dim]` and Indexer scoring keys `[G, index_head_dim]` —
//! device-resident with packed sign bits alongside, and selection runs
//! **two-stage**: training-free BDP sign-agreement recall over ALL entries
//! (cheap XNOR+popcount, no depth limit) shortlists top-M, then the learned
//! Indexer float score re-ranks only the shortlist for the exact top-k the
//! model was trained to expect. Both stages run fully on-device.
//!
//! Entries are pre-RoPE and position-free (`pos` carries each group's start
//! for RoPE-at-read in the attention kernel).

use candle::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
use super::scatter::{rows_scatter, RowRun};
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::Generation;

/// Words per packed sign row.
fn sign_words(dim: usize) -> usize {
    dim.div_ceil(32)
}

#[cfg(all(test, feature = "cuda"))]
mod score_reduce_tests {
    use super::indexer_score_reduce;
    use candle::{DType, Device, Result, Tensor};

    /// The fused Indexer score reduction against the eager chain it replaces.
    ///
    /// Bit-equality is NOT the bar, and the reason is measured rather than
    /// assumed: candle reduces the head axis with a tree while the kernel walks
    /// it sequentially, so the summation orders differ by construction (3 ULP at
    /// H = 64) and matching would mean replicating candle's reduction shape.
    /// Since they cannot agree, the kernel is held to being at least as faithful
    /// to exact arithmetic as the eager chain — the same bar the compressor pool
    /// uses, and the one that matters here because this feeds an argsort where a
    /// few ULP can reorder near-tied candidates. The kernel accumulates EXACTLY
    /// in f32 (two-product by FMA, two-sum compensation), so it should win
    /// outright — measured 0 relative error on every shape below. Note the bar
    /// is not decoration: plain Kahan compensation measured WORSE than the eager
    /// chain here and failed this assert, because `relu` leaves every term
    /// non-negative while `w` is signed, so the running sum cancels toward zero
    /// while the products stay large. Do not weaken the summation to match a
    /// tolerance; the tolerance is the eager chain's own error.
    ///
    /// Covers both selectors' shapes: the decode wave's `[sessions, heads,
    /// shortlist]` and prefill's `[tokens, heads, corpus]`, masked and unmasked,
    /// including a row masked to ZERO valid columns and a row fully valid.
    /// Half the cases feed STRIDED scores — the pool kernel shipped with a
    /// packed-input assumption that only the model gate caught, so this covers
    /// that axis up front.
    #[test]
    #[ignore]
    fn indexer_score_reduce_matches_eager() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for &(b, h, m) in &[
            (16usize, 64usize, 128usize),
            (1, 64, 1024),
            (5, 32, 37),
            (128, 16, 512),
            (3, 1, 64),
        ] {
            let base = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
            let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
            // Strided on half the shapes: a channel-narrow of a wider block.
            let strided = m >= 128;
            let scores = if strided {
                let pad = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
                Tensor::cat(&[&base, &pad], 2)?.narrow(2, 0, m)?
            } else {
                base
            };
            assert_eq!(scores.is_contiguous(), !strided, "strided setup ({b},{h},{m})");

            // Counts spanning the interesting boundaries: none valid, all valid,
            // and a ragged middle.
            let cv: Vec<u32> = (0..b).map(|i| ((i * 7) % (m + 1)) as u32).collect();
            let counts = Tensor::from_vec(cv, b, &dev)?;

            for use_mask in [false, true] {
                let cnt = if use_mask { Some(&counts) } else { None };
                let got = indexer_score_reduce(&scores, &w, cnt)?;

                let want = {
                    let e = scores.relu()?.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
                    match cnt {
                        None => e,
                        Some(c) => {
                            let col =
                                Tensor::arange(0u32, m as u32, &dev)?.reshape((1, m))?;
                            let mask = col
                                .broadcast_lt(&c.reshape((b, 1))?)?
                                .to_dtype(DType::F32)?
                                .affine(1e30, -1e30)?;
                            e.broadcast_add(&mask)?
                        }
                    }
                };

                // The same expression in exact f64 on the host — the third
                // opinion that says which device path owns the divergence.
                let exact = {
                    let sc: Vec<f32> = scores.flatten_all()?.to_vec1()?;
                    let wv: Vec<f32> = w.flatten_all()?.to_vec1()?;
                    let cv: Vec<u32> = counts.flatten_all()?.to_vec1()?;
                    let mut o = vec![0f32; b * m];
                    for bi in 0..b {
                        for j in 0..m {
                            let mut s = 0f64;
                            for hi in 0..h {
                                let v = sc[(bi * h + hi) * m + j];
                                s += (v.max(0.0) as f64) * (wv[bi * h + hi] as f64);
                            }
                            let pad = use_mask && (j as u32) >= cv[bi];
                            o[bi * m + j] = s as f32 + if pad { -1e30 } else { 0.0 };
                        }
                    }
                    o
                };

                let a = want.flatten_all()?.to_vec1::<f32>()?;
                let x = got.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(a.len(), x.len(), "shape ({b},{h},{m}) mask={use_mask}");
                let rel = |p: f32, q: f32| {
                    if p == q {
                        0.0
                    } else {
                        (p - q).abs() / p.abs().max(q.abs()).max(1e-30)
                    }
                };
                let (mut k_err, mut e_err) = (0f32, 0f32);
                for ((p, q), z) in a.iter().zip(&x).zip(&exact) {
                    k_err = k_err.max(rel(*q, *z));
                    e_err = e_err.max(rel(*p, *z));
                }
                println!(
                    "[score b={b} h={h} m={m} mask={use_mask}] kernel-vs-exact \
                     {k_err:.3e}  eager-vs-exact {e_err:.3e}"
                );
                assert!(
                    k_err <= e_err.max(1e-7),
                    "kernel drifts from exact arithmetic MORE than the eager chain \
                     it replaces ({b},{h},{m}) mask={use_mask}: kernel {k_err:.3e} \
                     > eager {e_err:.3e}"
                );
            }
        }
        Ok(())
    }

    /// Isolation bench at the shapes both selectors actually run: the decode
    /// wave's `[sessions, idx_heads, shortlist]` and prefill's
    /// `[tokens, idx_heads, corpus]`.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     bench_indexer_score -- --ignored --nocapture
    ///
    /// The `--launch-skip` is REQUIRED and shape-dependent: this bench ramps the
    /// device for 100 launches and then runs 8 warm-up + 5 x 40 timed launches
    /// per shape, so profiling from launch 0 captures the ramp (which runs
    /// unmasked) rather than any shape the bench reports. 850 lands in the
    /// prefill shape, the last of the five:
    ///
    ///   ncu --kernel-name indexer_score_reduce_kernel \
    ///     --launch-skip 850 --launch-count 1 \
    ///     --section SpeedOfLight --section ComputeWorkloadAnalysis \
    ///     target/release/deps/candle_transformers-<hash>.exe \
    ///     bench_indexer_score --ignored --nocapture
    ///
    /// ComputeWorkloadAnalysis is not optional here — it names the saturated
    /// PIPE. SpeedOfLight alone said "compute-bound" while the actual finding
    /// was FP64 at 84.7% on a part that runs FP64 at 1/64 rate.
    #[test]
    #[ignore]
    fn bench_indexer_score() -> Result<()> {
        let dev = Device::new_cuda(0)?;

        // Ramp the device BEFORE the first timed shape, so module load and the
        // idle-clock climb land outside the measurement.
        {
            let s = Tensor::randn(0f32, 1.0, (512usize, 64usize, 512usize), &dev)?;
            let wv = Tensor::randn(0f32, 1.0, (512usize, 64usize), &dev)?;
            for _ in 0..100 {
                let _ = indexer_score_reduce(&s, &wv, None)?;
            }
            dev.synchronize()?;
        }

        // Timing is MIN-of-rounds, not a single mean, because this box is WDDM:
        // some round takes a ~60 us scheduling stall that has nothing to do with
        // the kernel, and a mean folds it into the result. It is identifiable as
        // interference rather than cost because it lands on ONE shape per run,
        // a different shape each run, and inflates the eager and fused sides
        // together (one run put decode x1 at 113.68 us fused against 108.71 us
        // eager, where that shape steady-states at 11 vs 70). The minimum over
        // several rounds is the shape's real cost; the stall shows up as spread
        // between rounds, not as the reported number.
        let best = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                dev.synchronize()?;
                let t = std::time::Instant::now();
                for _ in 0..40 {
                    f()?;
                }
                dev.synchronize()?;
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / 40.0);
            }
            Ok(best)
        };

        for &(b, h, m, tag) in &[
            (1usize, 64usize, 128usize, "decode x1"),
            (8, 64, 128, "decode x8"),
            (16, 64, 128, "decode x16"),
            (16, 64, 1024, "decode deep"),
            (512, 64, 512, "prefill"),
        ] {
            let scores = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
            let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
            let cv: Vec<u32> = (0..b).map(|i| ((i * 7) % (m + 1)) as u32).collect();
            let counts = Tensor::from_vec(cv, b, &dev)?;

            for _ in 0..8 {
                let _ = indexer_score_reduce(&scores, &w, Some(&counts))?;
            }
            let fused = best(&mut || {
                indexer_score_reduce(&scores, &w, Some(&counts))?;
                Ok(())
            })?;

            let eager = |s: &Tensor| -> Result<Tensor> {
                let e = s.relu()?.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
                let col = Tensor::arange(0u32, m as u32, &dev)?.reshape((1, m))?;
                let mask = col
                    .broadcast_lt(&counts.reshape((b, 1))?)?
                    .to_dtype(DType::F32)?
                    .affine(1e30, -1e30)?;
                e.broadcast_add(&mask)
            };
            for _ in 0..8 {
                let _ = eager(&scores)?;
            }
            let eg = best(&mut || {
                eager(&scores)?;
                Ok(())
            })?;

            // Bytes the kernel must move: scores once, w once, out once.
            let bytes = (b * h * m + b * h + b * m) * 4;
            println!(
                "[score {tag:>12}] b={b:<4} h={h:<3} m={m:<5} fused {fused:7.2} us  \
                 eager {eg:8.2} us  {:5.2}x   {:6.2} GB/s",
                eg / fused,
                bytes as f64 / (fused * 1e-6) / 1e9
            );
        }
        Ok(())
    }

    /// A row's reduced score must not depend on how many other rows shared the
    /// launch — the same batch-width invariance the compressor pool needs, and
    /// for the same reason: a decode wave reduces however many sessions selected
    /// this step, so if width perturbed the result a session's top-k would
    /// depend on who it was batched with.
    #[test]
    #[ignore]
    fn indexer_score_reduce_is_batch_width_invariant() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (b, h, m) = (24usize, 64usize, 256usize);
        let scores = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
        let cv: Vec<u32> = (0..b).map(|i| ((i * 11) % (m + 1)) as u32).collect();
        let counts = Tensor::from_vec(cv, b, &dev)?;
        let batched = indexer_score_reduce(&scores, &w, Some(&counts))?;
        for i in 0..b {
            let alone = indexer_score_reduce(
                &scores.narrow(0, i, 1)?,
                &w.narrow(0, i, 1)?,
                Some(&counts.narrow(0, i, 1)?),
            )?;
            let p = batched.narrow(0, i, 1)?.flatten_all()?.to_vec1::<f32>()?;
            let q = alone.flatten_all()?.to_vec1::<f32>()?;
            for (j, (x, y)) in p.iter().zip(&q).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "row {i} elem {j} changed with batch width: {x} vs {y}"
                );
            }
        }
        Ok(())
    }

    /// The two degenerate inputs the fused path could get wrong SILENTLY.
    ///
    /// `h == 0`: the reduction over zero heads is 0, and the CUDA path has to
    /// WRITE that. `out` is allocated uninitialised (hot-path invariant 6), so a
    /// launcher that bails on an empty head axis hands whatever was in that VRAM
    /// straight to the argsort — no error, and a different wrong answer each
    /// run. The eager path returns zeros, so the two must agree.
    ///
    /// Strided `counts`: every other input is read through its own stride, so
    /// counts must be too. A column of a `[b, 2]` tensor has stride 2, and
    /// reading it as packed silently takes the NEIGHBOURING column as a
    /// visibility bound — masking real entries out of the top-k, or letting
    /// padding in, with nothing raised.
    #[test]
    #[ignore]
    fn degenerate_head_axis_and_strided_counts() -> Result<()> {
        let dev = Device::new_cuda(0)?;

        // h == 0, masked and unmasked.
        let (b, m) = (3usize, 16usize);
        let scores = Tensor::zeros((b, 0, m), DType::F32, &dev)?;
        let w = Tensor::zeros((b, 0), DType::F32, &dev)?;
        let cv: Vec<u32> = vec![0, 7, m as u32];
        let counts = Tensor::from_vec(cv.clone(), b, &dev)?;
        for cnt in [None, Some(&counts)] {
            let got = indexer_score_reduce(&scores, &w, cnt)?;
            assert_eq!(got.dims(), [b, m], "h=0 shape");
            let v = got.flatten_all()?.to_vec1::<f32>()?;
            for (i, x) in v.iter().enumerate() {
                let (row, col) = (i / m, i % m);
                let want = match cnt {
                    Some(_) if (col as u32) >= cv[row] => -1e30f32,
                    _ => 0f32,
                };
                assert_eq!(
                    x.to_bits(),
                    want.to_bits(),
                    "h=0 row {row} col {col}: got {x}, want {want} \
                     (uninitialised output?)"
                );
            }
        }

        // Strided counts: a column of a [b, 2] block, stride 2, against the
        // same values passed packed.
        let (b, h, m) = (6usize, 8usize, 32usize);
        let scores = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
        let want_v: Vec<u32> = (0..b).map(|i| ((i * 5) % (m + 1)) as u32).collect();
        // Interleave with a decoy column that would produce a DIFFERENT mask if
        // the kernel read this as packed.
        let mut inter: Vec<u32> = Vec::with_capacity(b * 2);
        for (i, &c) in want_v.iter().enumerate() {
            inter.push(c);
            inter.push(((i * 3 + 1) % (m + 1)) as u32);
        }
        let strided = Tensor::from_vec(inter, (b, 2), &dev)?
            .narrow(1, 0, 1)?
            .squeeze(1)?;
        assert_eq!(strided.dims(), [b], "strided counts shape");
        assert!(!strided.is_contiguous(), "counts should be strided");
        let packed = Tensor::from_vec(want_v, b, &dev)?;

        let a = indexer_score_reduce(&scores, &w, Some(&strided))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let e = indexer_score_reduce(&scores, &w, Some(&packed))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (x, y)) in a.iter().zip(&e).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "strided counts elem {i} ({}, {}) diverged: {x} vs {y}",
                i / m,
                i % m
            );
        }
        Ok(())
    }
}

/// `out[b, j] = Σ_h relu(scores[b, h, j])·w[b, h] + (j < counts[b] ? 0 : -1e30)`
/// — the two-stage selection's precision-stage reduction, for `scores`
/// `[B, H, M]` and per-head gate weights `w` `[B, H]`.
///
/// On CUDA this is ONE launch; eagerly it is eight (`relu`, `broadcast_mul`, a
/// `sum` over the head axis — a non-last dim, so candle's generic reduce — then
/// `arange`, `broadcast_lt`, `to_dtype`, `affine` to build the padding mask
/// on-device, and `broadcast_add` to apply it), and eight is a floor since that
/// `sum` may itself be more than one. Both batched selectors share it: the decode wave's
/// [`two_stage_select_batched`] and prefill's
/// [`FloatGallery::batched_causal_select_device`], which differ only in what
/// `counts` means — a per-session shortlist width in one, a per-token causal
/// visibility bound in the other.
///
/// `counts = None` leaves every column unmasked. The eager form below is the CPU
/// path and the reference `indexer_score_reduce_matches_eager` compares against.
///
/// Scope is deliberately narrow: `two_stage_select_batched` records a measured
/// −29% prefill from fusing this stage's MATMUL and ARGSORT, both of which want
/// the whole GPU. This replaces only the elementwise reduction between them.
fn indexer_score_reduce(
    scores: &Tensor,
    w: &Tensor,
    counts: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, h, m) = scores.dims3()?;
    if w.dims2()? != (b, h) {
        candle::bail!(
            "indexer_score_reduce: scores {:?} and weights {:?} disagree",
            scores.dims(),
            w.dims()
        );
    }

    #[cfg(feature = "cuda")]
    if matches!(scores.device(), Device::Cuda(_)) {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        use candle_kernels::simple::indexer_score::run_indexer_score_reduce;

        let dev = match scores.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!("guarded by the matches! above"),
        };
        let stream = dev.cuda_stream();
        // The kernel writes every element, so allocate uninitialised rather than
        // memset bytes it is about to stamp (hot-path invariant 6).
        let out = Tensor::empty((b, m), DType::F32, scores.device())?;
        {
            let (s_sc, l_sc) = scores.storage_and_layout();
            let (s_w, l_w) = w.storage_and_layout();
            let (s_out, _) = out.storage_and_layout();
            let f32_ptr = |st: &Storage, off: usize| -> Result<u64> {
                match st {
                    Storage::Cuda(c) => {
                        Ok(c.as_cuda_slice::<f32>()?.slice(off..).device_ptr(&stream).0)
                    }
                    _ => candle::bail!("indexer_score_reduce: expected CUDA storage"),
                }
            };
            // `counts` is u32 and optional; a null pointer means "no padding".
            let cnt_guard = match counts {
                Some(c) => {
                    if c.dims1()? != b {
                        candle::bail!(
                            "indexer_score_reduce: counts {:?} must be [{b}]",
                            c.dims()
                        );
                    }
                    Some(c.storage_and_layout())
                }
                None => None,
            };
            // Stride threaded like every other input: `counts` is the one a
            // caller is most likely to pass as a view of something wider, and
            // assuming stride 1 would read a neighbour as a visibility bound.
            let (p_cnt, cnt_s) = match &cnt_guard {
                Some((st, lay)) => match &**st {
                    Storage::Cuda(c) => (
                        c.as_cuda_slice::<u32>()?
                            .slice(lay.start_offset()..)
                            .device_ptr(&stream)
                            .0,
                        lay.stride()[0] as i64,
                    ),
                    _ => candle::bail!("indexer_score_reduce: expected CUDA counts"),
                },
                None => (0u64, 1i64),
            };
            let p_sc = f32_ptr(&s_sc, l_sc.start_offset())?;
            let p_w = f32_ptr(&s_w, l_w.start_offset())?;
            let p_out = f32_ptr(&s_out, 0)?;
            let (ss, ws) = (l_sc.stride(), l_w.stride());
            unsafe {
                run_indexer_score_reduce(
                    p_sc as *const f32,
                    p_w as *const f32,
                    p_cnt as *const u32,
                    p_out as *mut f32,
                    b as i32,
                    h as i32,
                    m as i32,
                    ss[0] as i64,
                    ss[1] as i64,
                    ss[2] as i64,
                    ws[0] as i64,
                    ws[1] as i64,
                    cnt_s,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
        }
        return Ok(out);
    }

    let weighted = scores.relu()?.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
    match counts {
        None => Ok(weighted),
        Some(c) => {
            let col = Tensor::arange(0u32, m as u32, scores.device())?.reshape((1, m))?;
            let mask = col
                .broadcast_lt(&c.reshape((b, 1))?)?
                .to_dtype(DType::F32)?
                .affine(1e30, -1e30)?; // valid → 0, pad → −1e30
            weighted.broadcast_add(&mask)
        }
    }
}

/// Entry count past which the HOT tier — the position-free two-region cache
/// (`nope_i8`/`nope_scale`/`rope_bf`) plus the Indexer `keys` — spills from the
/// GPU (hot) to CPU RAM (warm). The `signs` + `pos` index stays GPU-resident at
/// any depth (the BDP scan reads all of it, and it is tiny — `sign_words·4 + 4`
/// bytes/entry). Below the threshold a gallery is fully hot, so short
/// conversations and the reference paths pay nothing; beyond it the resident
/// footprint is bounded to the index while the corpus grows in RAM and the
/// bounded selection is gathered back per query (int8, ~576 B/entry).
/// At 8192 entries (`ratio 4` ⇒ ~32k tokens) the hot cache is ≤4.7 MB/gallery.
const HOT_ENTRY_CAP: usize = 8192;

/// Compressed-corpus store with a packed sign index. The `signs`/`pos` index is
/// always GPU-resident; the HOT two-region cache + `keys` spill to CPU RAM past
/// [`HOT_ENTRY_CAP`] entries (`spilled`), keeping the resident footprint bounded
/// at unbounded depth (§L). Grows by doubling.
///
/// The two-region cache (`nope_i8`/`nope_scale`/`rope_bf`) IS the canonical
/// stored latent — there is no separate f32 archive. Reference/bench callers
/// that want the dense f32 rows reconstruct them from the two-region
/// ([`Self::two_region_rows_f32`]), i.e. exactly what the decode/prefill kernels
/// read; the live attention path reads the two-region directly and never
/// materialises f32.
pub struct FloatGallery {
    keys: Tensor, // [cap, index_head_dim] f32, pre-RoPE — GPU while hot, CPU when spilled
    // The HOT retrieval artifact: the position-free two-region cache, built from
    // the incoming rows on append. The decode/prefill readers rotate the rope
    // bands at read time. GPU while hot, CPU when spilled (re-heated per query by
    // `gather_corpus`).
    nope_i8: Tensor,    // [cap, nope_dim] u8  (nope int8)
    nope_scale: Tensor, // [cap, nope_bands] f32 (per-nope-band amax)
    rope_bf: Tensor,    // [cap, rope_dim] bf16 (rope pre-rotation)
    signs: Tensor,      // [cap, sign_words] u32 — always GPU
    pos: Tensor,        // [cap] u32 group-start positions — always GPU
    len: usize,
    cap: usize,
    head_dim: usize,
    index_head_dim: usize,
    device: Device,
    /// True once the two-region cache + `keys` have moved to CPU RAM (warm tier).
    spilled: bool,
    /// Cached device base addresses of the four hot-cache regions
    /// (`nope_i8`, `nope_scale`, `rope_bf`, `pos`) for the batched gather's
    /// pointer table. Invalidated (`None`) whenever those tensors realloc
    /// (`grow_to`) or move off the GPU (`maybe_spill`), so a stale address is
    /// never handed to the kernel.
    region_ptr_cache: std::cell::Cell<Option<[u64; 4]>>,
}

/// Latent geometry for the two-region cache (single-latent DeepSeek-V4).
const NOPE_DIM: usize = 448;
const ROPE_DIM: usize = 64;
const SUB_DIM: usize = 32;
const NOPE_BANDS: usize = NOPE_DIM / SUB_DIM; // 14

impl FloatGallery {
    pub fn new(
        device: &Device,
        head_dim: usize,
        index_head_dim: usize,
        initial_cap: usize,
    ) -> Result<Self> {
        // The two-region cache builder + readers are specialized to the
        // single-latent geometry (`HEAD_DIM = NOPE_DIM + ROPE_DIM = 512`): the
        // buffers below are sized by those constants and the build/decode
        // kernels are compiled at `<512, 64>`. A gallery of any other `head_dim`
        // would silently mis-stride, so reject it up front.
        if head_dim != NOPE_DIM + ROPE_DIM {
            candle::bail!(
                "FloatGallery head_dim {} must equal NOPE_DIM+ROPE_DIM ({}) for the two-region corpus cache",
                head_dim,
                NOPE_DIM + ROPE_DIM,
            );
        }
        let cap = initial_cap.max(1);
        Ok(Self {
            keys: Tensor::zeros((cap, index_head_dim), DType::F32, device)?,
            nope_i8: Tensor::zeros((cap, NOPE_DIM), DType::U8, device)?,
            nope_scale: Tensor::zeros((cap, NOPE_BANDS), DType::F32, device)?,
            rope_bf: Tensor::zeros((cap, ROPE_DIM), DType::BF16, device)?,
            signs: Tensor::zeros((cap, sign_words(index_head_dim)), DType::U32, device)?,
            pos: Tensor::zeros(cap, DType::U32, device)?,
            len: 0,
            cap,
            head_dim,
            index_head_dim,
            device: device.clone(),
            spilled: false,
            region_ptr_cache: std::cell::Cell::new(None),
        })
    }

    /// Whether the float corpus pair currently lives in CPU RAM (warm tier).
    pub fn is_spilled(&self) -> bool {
        self.spilled
    }

    /// The device the spillable tier — the two-region cache + Indexer `keys` —
    /// currently lives on (CPU once spilled). `signs`/`pos` are always on
    /// [`Self::device`].
    fn float_device(&self) -> Device {
        if self.spilled {
            Device::Cpu
        } else {
            self.device.clone()
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Roll the gallery back to its first `len` entries. Every reader narrows to
    /// `self.len` and [`Self::append_batch`] writes rows at `len..`, so entries
    /// beyond the new length become inert and are overwritten by the next
    /// append — no buffer surgery needed. Used by the speculative-decode verify
    /// rollback to drop groups pooled over rejected draft tokens.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// The live attended-entry rows `[len, head_dim]` f32, reconstructed from the
    /// two-region cache (reference/bench only — the live path reads the
    /// two-region directly). Returns exactly what the decode/prefill kernels see.
    pub fn attn_entries(&self) -> Result<Tensor> {
        let n = self.len.max(1);
        self.two_region_rows_f32(&self.nope_i8, &self.nope_scale, &self.rope_bf, 0, n)
    }

    /// The live position view `[len]` (`comp_pos`).
    pub fn positions(&self) -> Result<Tensor> {
        self.pos.narrow(0, 0, self.len.max(1))
    }

    /// The live scoring-key view `[len, index_head_dim]`.
    pub fn scoring_keys(&self) -> Result<Tensor> {
        self.keys.narrow(0, 0, self.len.max(1))
    }

    /// The live packed-sign view `[len, words]`.
    pub fn packed_signs(&self) -> Result<Tensor> {
        self.signs.narrow(0, 0, self.len.max(1))
    }

    fn grow_to(&mut self, need: usize) -> Result<()> {
        if need <= self.cap {
            return Ok(());
        }
        let mut new_cap = self.cap;
        while new_cap < need {
            new_cap *= 2;
        }
        // The float pair grows on whichever tier it currently lives on; the
        // sign/pos index always grows on the GPU.
        let fdev = self.float_device();
        let grow = |t: &Tensor, cols: usize, dtype: DType, dev: &Device| -> Result<Tensor> {
            let nt = if cols == 0 {
                Tensor::zeros(new_cap, dtype, dev)?
            } else {
                Tensor::zeros((new_cap, cols), dtype, dev)?
            };
            nt.slice_set(t, 0, 0)?;
            Ok(nt)
        };
        self.keys = grow(&self.keys, self.index_head_dim, DType::F32, &fdev)?;
        // The two-region cache lives on the spillable tier alongside `keys`
        // (GPU while hot, CPU once spilled).
        self.nope_i8 = grow(&self.nope_i8, NOPE_DIM, DType::U8, &fdev)?;
        self.nope_scale = grow(&self.nope_scale, NOPE_BANDS, DType::F32, &fdev)?;
        self.rope_bf = grow(&self.rope_bf, ROPE_DIM, DType::BF16, &fdev)?;
        self.signs = grow(
            &self.signs,
            sign_words(self.index_head_dim),
            DType::U32,
            &self.device,
        )?;
        self.pos = grow(&self.pos, 0, DType::U32, &self.device)?;
        self.cap = new_cap;
        self.region_ptr_cache.set(None); // regions reallocated
        Ok(())
    }

    /// Move the HOT tier — the two-region cache + Indexer `keys` — from GPU to
    /// CPU RAM once the entry count crosses [`HOT_ENTRY_CAP`] (`signs`/`pos` stay
    /// GPU). One-way: the corpus only grows,
    /// and re-heating the whole spilled tier would defeat the bound — per-query
    /// gathers pull the bounded working set back instead.
    fn maybe_spill(&mut self, prospective_len: usize) -> Result<()> {
        if self.spilled || prospective_len <= HOT_ENTRY_CAP || !self.device.is_cuda() {
            return Ok(());
        }
        self.keys = self.keys.to_device(&Device::Cpu)?;
        self.nope_i8 = self.nope_i8.to_device(&Device::Cpu)?;
        self.nope_scale = self.nope_scale.to_device(&Device::Cpu)?;
        self.rope_bf = self.rope_bf.to_device(&Device::Cpu)?;
        self.spilled = true;
        self.region_ptr_cache.set(None); // regions moved to CPU
        Ok(())
    }

    /// Append `n` completed groups: attended rows `[n, head_dim]` f32, scoring
    /// keys `[n, index_head_dim]` f32 (both pre-RoPE, device), and their
    /// group-start positions. Sign bits are packed on-device.
    ///
    /// One gallery's [`append_batch_all`] — the reference/streamed form. The
    /// wave never calls this: it appends every session's groups at once.
    #[cfg(feature = "cuda")]
    pub fn append_batch(
        &mut self,
        attn_rows: &Tensor,
        key_rows: &Tensor,
        positions: &[u32],
    ) -> Result<()> {
        let n = attn_rows.dim(0)?;
        if n == 0 {
            return Ok(());
        }
        let scope = crate::models::deepseek4::desc::scope(&self.device)?;
        append_batch_all(
            &mut [AppendSlice {
                gallery: self,
                runs: vec![(0, n)],
            }],
            attn_rows,
            key_rows,
            positions,
            &scope,
        )
    }

    /// Gather the dense f32 rows and positions for `gids` (absolute entry ids,
    /// GPU u32) into a COMPACTED GPU pair — `(attn [k, head_dim], pos [k])`.
    /// Reconstructs the rows from the compacted two-region (`gather_corpus` is
    /// tier-aware), so it returns exactly what the decode/prefill kernels read.
    /// Reference/rebuild callers only — the hot path is `gather_corpus`.
    #[cfg(feature = "cuda")]
    pub fn gather_selected(&self, gids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (ni8, nsc, rbf, pos) = self.gather_corpus(gids)?;
        let k = pos.dim(0)?;
        let attn = self.two_region_rows_f32(&ni8, &nsc, &rbf, 0, k)?;
        Ok((attn.contiguous()?, pos.contiguous()?))
    }

    /// Reconstruct dense f32 latent rows `[n, head_dim]` from `n` two-region rows
    /// (`nope_i8`/`nope_scale`/`rope_bf`, starting at `lo`) — the exact inverse
    /// of `build_corpus_cache_into`: nope band value `= int8 · per-band scale`,
    /// rope band `= bf16 → f32`. Returns what the kernels read. Host-side
    /// (reference/bench only — the live attention path reads the two-region
    /// directly and never materialises f32); the result lands on the gallery
    /// device.
    fn two_region_rows_f32(
        &self,
        nope_i8: &Tensor,
        nope_scale: &Tensor,
        rope_bf: &Tensor,
        lo: usize,
        n: usize,
    ) -> Result<Tensor> {
        let ni8: Vec<u8> = nope_i8.narrow(0, lo, n)?.flatten_all()?.to_vec1::<u8>()?;
        let nsc: Vec<f32> = nope_scale
            .narrow(0, lo, n)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rbf: Vec<f32> = rope_bf
            .narrow(0, lo, n)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let hd = self.head_dim;
        let mut out = vec![0f32; n * hd];
        for i in 0..n {
            for band in 0..NOPE_BANDS {
                let scale = nsc[i * NOPE_BANDS + band];
                for j in 0..SUB_DIM {
                    let d = band * SUB_DIM + j;
                    out[i * hd + d] = (ni8[i * NOPE_DIM + d] as i8 as f32) * scale;
                }
            }
            for j in 0..ROPE_DIM {
                out[i * hd + NOPE_DIM + j] = rbf[i * ROPE_DIM + j];
            }
        }
        Tensor::from_vec(out, (n, hd), &self.device)
    }

    /// Gather the HOT two-region cache for `gids` into a COMPACTED GPU tuple —
    /// `(nope_i8 [k, NOPE_DIM], nope_scale [k, NOPE_BANDS], rope_bf [k, ROPE_DIM],
    /// pos [k])`. This is what decode/prefill attend. Tier-aware: in place on the
    /// GPU while hot, re-heated from CPU RAM when spilled — touching only the `k`
    /// selected rows (~576 B each), so the transfer is bounded at any depth. `pos`
    /// is always GPU-resident and gathered in place.
    #[cfg(feature = "cuda")]
    pub fn gather_corpus(&self, gids: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let k = gids.dim(0)?;
        // Fresh destinations, then the SAME gather that fills a shared block —
        // one implementation of the tier logic, not two that must be kept in
        // step. The kernel (or, when spilled, the scatter) writes every row, so
        // these are uninitialised (hot-path invariant 6).
        let out_nope = Tensor::empty((k.max(1), NOPE_DIM), DType::U8, &self.device)?;
        let out_scale = Tensor::empty((k.max(1), NOPE_BANDS), DType::F32, &self.device)?;
        let out_rope = Tensor::empty((k.max(1), ROPE_DIM), DType::BF16, &self.device)?;
        let out_pos = Tensor::empty(k.max(1), DType::U32, &self.device)?;
        let scope = crate::models::deepseek4::desc::scope(&self.device)?;
        self.gather_corpus_into(gids, &out_nope, &out_scale, &out_rope, &out_pos, 0, &scope)?;
        if k == 0 {
            return Ok((
                out_nope.narrow(0, 0, 0)?,
                out_scale.narrow(0, 0, 0)?,
                out_rope.narrow(0, 0, 0)?,
                out_pos.narrow(0, 0, 0)?,
            ));
        }
        Ok((out_nope, out_scale, out_rope, out_pos))
    }

    /// Gather this session's `k` selected hot-cache rows (`gids`) DIRECTLY into a
    /// shared, pre-allocated output block at row `row_offset` — `out_nope`
    /// `[total, NOPE_DIM]` u8, `out_scale` `[total, NOPE_BANDS]` f32, `out_rope`
    /// `[total, ROPE_DIM]` bf16, `out_pos` `[total]` u32. One fused launch (all
    /// four regions) replaces the four per-region `index_select`s; writing in
    /// place replaces the cross-session `cat`. Tier-aware: the GPU kernel gathers
    /// hot galleries; a spilled gallery re-heats its `k` rows from CPU RAM and
    /// `slice_set`s them in (bounded transfer, as before).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn gather_corpus_into(
        &self,
        gids: &Tensor,
        out_nope: &Tensor,
        out_scale: &Tensor,
        out_rope: &Tensor,
        out_pos: &Tensor,
        row_offset: usize,
        generation: &Generation,
    ) -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let k = gids.dim(0)?;
        if k == 0 {
            return Ok(());
        }
        let live = self.len.max(1);
        if self.spilled {
            // Warm tier. The float pair lives in PAGEABLE host RAM, which no
            // kernel can address, so the row gather itself must run on the host —
            // the one place invariant 4's "no host compute" does not reach,
            // because there is no kernel that could do it. (What would change
            // that is the pinned warm pool of docs/kv_tier_migration.md: page-
            // locked and device-mapped, the hot path's fused gather would read
            // this tier in place. It is a bounded POOL by design — page-locking
            // every spilled arena wholesale would make an unbounded amount of
            // host memory unswappable.)
            //
            // What the host gather does NOT have to be is a copy per region. It
            // packs all three regions into ONE staging buffer, so the re-heat is
            // a single transfer and a single scatter instead of three uploads and
            // four `slice_set` launches. `pos` never spills, so its rows are
            // gathered on the device and ride the same scatter.
            let ids: Vec<u32> = gids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
            let regions = [&self.nope_i8, &self.nope_scale, &self.rope_bf];
            let widths: Vec<usize> = regions
                .iter()
                .map(|t| Ok(t.dim(1)? * t.dtype().size_in_bytes()))
                .collect::<Result<_>>()?;
            let mut staged: Vec<u8> = Vec::with_capacity(k * widths.iter().sum::<usize>());
            for (t, w) in regions.iter().zip(&widths) {
                pack_rows(t, &ids, live, *w, &mut staged)?;
            }
            let n_bytes = staged.len();
            let staged = Tensor::from_vec(staged, n_bytes, &Device::Cpu)?.to_device(&self.device)?;

            let pos = self.pos.narrow(0, 0, live)?.index_select(gids, 0)?;
            let mut runs = vec![RowRun::new(pos, out_pos, row_offset)];
            let mut off = 0usize;
            for ((w, dst), _) in widths
                .iter()
                .zip([out_nope, out_scale, out_rope])
                .zip(regions.iter())
            {
                // A byte block reshaped to this region's row width — the scatter
                // matches rows by BYTE width, not element type.
                runs.push(RowRun::new(
                    staged.narrow(0, off, k * w)?.reshape((k, *w))?,
                    dst,
                    row_offset,
                ));
                off += k * w;
            }
            rows_scatter(&runs, generation)?;
            return Ok(());
        }
        // Hot tier: one fused gather launch. The four source regions and the four
        // outputs are all contiguous (dim-0 prefixes / fresh allocations), so
        // their storage pointers address logical row 0; the kernel reinterprets
        // each row as 32-bit words (all widths are 4-byte multiples).
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => candle::bail!("gather_corpus_into requires CUDA"),
        };
        let stream = dev.cuda_stream();
        let nope_src = self.nope_i8.narrow(0, 0, live)?;
        let scale_src = self.nope_scale.narrow(0, 0, live)?;
        let rope_src = self.rope_bf.narrow(0, 0, live)?;
        let pos_src = self.pos.narrow(0, 0, live)?;
        let gids = gids.contiguous()?;
        macro_rules! ptr_u32 {
            ($t:expr, $ty:ty) => {{
                let (s, _) = $t.storage_and_layout();
                match &*s {
                    Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?.device_ptr(&stream).0,
                    _ => candle::bail!("gather_corpus_into: non-CUDA storage"),
                }
            }};
        }
        let np = ptr_u32!(nope_src, u8);
        let sc = ptr_u32!(scale_src, f32);
        let rp = ptr_u32!(rope_src, half::bf16);
        let ps = ptr_u32!(pos_src, u32);
        let gp = ptr_u32!(gids, u32);
        let onp = ptr_u32!(out_nope, u8);
        let osc = ptr_u32!(out_scale, f32);
        let orp = ptr_u32!(out_rope, half::bf16);
        let ops = ptr_u32!(out_pos, u32);
        let code = unsafe {
            candle_kernels::simple::corpus_gather::run_corpus_gather_rows(
                np as *const u32,
                sc as *const u32,
                rp as *const u32,
                ps as *const u32,
                gp as *const u32,
                onp as *mut u32,
                osc as *mut u32,
                orp as *mut u32,
                ops as *mut u32,
                k as i32,
                row_offset as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("corpus_gather_rows launch failed: cuda error {code}");
        }
        Ok(())
    }

    /// The four hot-cache region device base addresses (`nope_i8`, `nope_scale`,
    /// `rope_bf`, `pos`), cached — the batched gather's per-session pointer table
    /// entry. Addresses are the tensors' row-0 storage bases (valid while hot and
    /// un-reallocated); the cache is invalidated on `grow_to`/`maybe_spill`.
    #[cfg(feature = "cuda")]
    fn hot_region_ptrs(&self) -> Result<[u64; 4]> {
        if let Some(p) = self.region_ptr_cache.get() {
            return Ok(p);
        }
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let stream = match &self.device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => candle::bail!("hot_region_ptrs requires CUDA"),
        };
        macro_rules! base {
            ($t:expr, $ty:ty) => {{
                let (s, _) = $t.storage_and_layout();
                match &*s {
                    Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?.device_ptr(&stream).0,
                    _ => candle::bail!("hot_region_ptrs requires CUDA"),
                }
            }};
        }
        let p = [
            base!(self.nope_i8, u8),
            base!(self.nope_scale, f32),
            base!(self.rope_bf, half::bf16),
            base!(self.pos, u32),
        ];
        self.region_ptr_cache.set(Some(p));
        Ok(p)
    }

    /// Gather the scoring `keys` for `ids` (GPU u32) onto the GPU — from CPU
    /// RAM when spilled, in place otherwise. Used by the two-stage rescore over
    /// the bounded shortlist.
    #[cfg(feature = "cuda")]
    fn gather_keys(&self, ids: &Tensor) -> Result<Tensor> {
        if self.spilled {
            let ids_cpu = ids.to_device(&Device::Cpu)?;
            self.keys
                .narrow(0, 0, self.len.max(1))?
                .index_select(&ids_cpu, 0)?
                .to_device(&self.device)
        } else {
            self.scoring_keys()?.index_select(ids, 0)
        }
    }

    /// Two-stage selection: BDP sign-agreement recall over all `len` entries →
    /// top-`top_m` shortlist → Indexer float re-score → top-`top_k`, returned
    /// **ascending** (the kernel's `comp_idx` order) as a `[k]` u32 device
    /// tensor plus the count.
    ///
    /// `q_idx` `[n_idx_heads, index_head_dim]` f32 and `weights`
    /// `[n_idx_heads]` f32 are the Indexer's per-head query vectors and gate
    /// weights (any positive per-head scale folded in by the caller — the
    /// score is `Σ_h relu(q_h·k)·w_h`, matching `Indexer` semantics).
    #[cfg(feature = "cuda")]
    pub fn two_stage_select(
        &self,
        q_idx: &Tensor,
        weights: &Tensor,
        top_m: usize,
        top_k: usize,
    ) -> Result<(Tensor, usize)> {
        self.two_stage_select_causal(q_idx, weights, top_m, top_k, self.len)
    }

    /// As [`Self::two_stage_select`] but restricted to the first `n_causal`
    /// entries — the causal-visibility bound a batched prefill needs when the
    /// whole prompt's corpus is appended up front: a query at prompt position `p`
    /// sees only the groups completed by `p`. Selecting over `[0, n_causal)` is
    /// bit-identical to running against a gallery that held exactly that many
    /// entries (each entry's sign-pack / key / score is independent of the ones
    /// after it), so it reproduces the per-token incremental path exactly.
    #[cfg(feature = "cuda")]
    pub fn two_stage_select_causal(
        &self,
        q_idx: &Tensor,
        weights: &Tensor,
        top_m: usize,
        top_k: usize,
        n_causal: usize,
    ) -> Result<(Tensor, usize)> {
        let n = n_causal.min(self.len);
        if n == 0 || top_k == 0 {
            return Ok((Tensor::zeros(1, DType::U32, &self.device)?, 0));
        }
        // The bitonic argsort caps at 1024 columns, which bounds the rescore
        // width — the recall shortlist itself is selected by the exact
        // histogram top-M (any corpus size).
        let m = top_m.clamp(1, 1024).min(n);

        // Stage 1 — recall: shortlist by packed-sign agreement over `[0, n)`.
        let shortlist: Tensor = if m >= n {
            Tensor::arange(0u32, n as u32, &self.device)?
        } else {
            let n_heads = q_idx.dim(0)?;
            let q_signs = sign_pack(q_idx)?;
            let signs = self.signs.narrow(0, 0, n)?;
            let counts = bdp_recall(&q_signs, &signs, self.index_head_dim)?;
            topm_select(&counts, m, n_heads * self.index_head_dim + 1)?
        };

        // Stage 2 — precision: Indexer float score over the shortlist only.
        // The keys gather is tier-aware — from CPU RAM when spilled, in place
        // otherwise — and touches only the bounded shortlist. The rescore stays
        // a cuBLAS matmul + a parallel argsort: both use the whole GPU, so a
        // single-block "fusion" of this stage is a net LOSS (measured -29%
        // prefill) — the launches here are few and each is already efficient.
        let keys = self.gather_keys(&shortlist)?; // [m, ih]
        let scores = q_idx.matmul(&keys.t()?.contiguous()?)?; // [h, m]
        let scores = scores.relu()?;
        let weighted = scores.broadcast_mul(&weights.reshape(((), 1))?)?.sum(0)?; // [m]
        let k = top_k.min(shortlist.dim(0)?);
        let order = weighted
            .unsqueeze(0)?
            .arg_sort_last_dim(false)?
            .squeeze(0)?;
        let picked = order.narrow(0, 0, k)?.contiguous()?; // shortlist-relative
        let gids = shortlist.index_select(&picked, 0)?; // absolute entry ids
                                                        // Ascending entry order: entry ids are exact in f32 (< 2^24), so an
                                                        // f32 argsort yields the ascending permutation on-device.
        let asc = gids
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .arg_sort_last_dim(true)?
            .squeeze(0)?
            .contiguous()?;
        let gids_sorted = gids.index_select(&asc, 0)?;
        Ok((gids_sorted.contiguous()?, k))
    }

    /// Whole-prompt batched causal select, **fully on-device**: score every
    /// prompt token's roped query against the corpus and take each token's causal
    /// top-k, returning the selection as GPU tensors — no `to_vec2`/host-sort/union
    /// round-trip (the readback that dominated `pprep:select`). `q_idx`
    /// `[s, n_idx_heads, index_head_dim]` are the roped per-token queries,
    /// `weights` `[s, n_idx_heads]` the gate weights, `n_visible[t]` the number of
    /// causally-visible entries for token `t` (`≤ len`).
    ///
    /// Returns `(comp_idx [s, kmax] u32, comp_cnt [s] u32, n_corpus)`:
    /// `comp_idx[t]` holds token `t`'s selected **absolute** entry ids ascending,
    /// `u32::MAX`-padded past `comp_cnt[t]`; the caller gathers the whole visible
    /// corpus (`0..n_corpus`) so those absolute ids index the gather directly.
    /// Bit-identical selection to the per-token [`Self::two_stage_select_causal`]
    /// loop in the in-regime case (widest window ≤ shortlist ⇒ two-stage == full
    /// top-k), gated by `batched_causal_select_matches_per_token`.
    ///
    /// **Regime:** valid ONLY when the widest causal window fits the shortlist
    /// (`max(n_visible) ≤ top_m ≤ 1024` at the call site — the device argsort caps
    /// at 1024 columns), so the gallery is never spilled (hot-only: reads
    /// `scoring_keys()` in place).
    #[cfg(feature = "cuda")]
    pub fn batched_causal_select_device(
        &self,
        q_idx: &Tensor,
        weights: &Tensor,
        n_visible: &[usize],
        top_k: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        debug_assert!(
            !self.spilled,
            "batched_causal_select_device is hot-only (regime-gated below HOT_ENTRY_CAP)"
        );
        let s = q_idx.dim(0)?;
        let h = q_idx.dim(1)?;
        let ih = self.index_head_dim;
        let n_corpus = n_visible.iter().copied().max().unwrap_or(0).min(self.len);
        if n_corpus == 0 || top_k == 0 {
            let idx = Tensor::full(u32::MAX, (s, 1), &self.device)?;
            let cnt = Tensor::zeros(s, DType::U32, &self.device)?;
            return Ok((idx, cnt, 0));
        }
        // scores[t,h,g] = relu(q_th · k_g); weighted[t,g] = Σ_h scores · w_th.
        let keys = self.scoring_keys()?.narrow(0, 0, n_corpus)?; // [n_corpus, ih]
        let scores = q_idx
            .reshape((s * h, ih))?
            .matmul(&keys.t()?.contiguous()?)? // [s*h, n_corpus]
            .reshape((s, h, n_corpus))?;
        // relu, weight, sum over heads and the CAUSAL mask in ONE launch: column
        // g is invalid for token t once g ≥ n_visible[t], and a masked entry
        // takes −1e30 so it never enters any top-k. Only the tiny [s] visibility
        // vector is uploaded.
        let vis: Vec<u32> = n_visible
            .iter()
            .map(|&nv| nv.min(n_corpus) as u32)
            .collect();
        let vis = Tensor::from_vec(vis, s, &self.device)?;
        let weighted = indexer_score_reduce(&scores, weights, Some(&vis))?; // [s, n_corpus]
        // Descending argsort → each token's gids by score; the column index IS the
        // absolute entry id (keys are `[0, n_corpus)` in order).
        let kmax = top_k.min(n_corpus);
        let order = weighted
            .arg_sort_last_dim(false)?
            .narrow(1, 0, kmax)?
            .contiguous()?; // [s, kmax] u32 — gids by score, desc
                            // Per-token valid count = min(top_k, n_visible[t], n_corpus).
        let cnt_v: Vec<u32> = n_visible
            .iter()
            .map(|&nv| top_k.min(nv.min(n_corpus)) as u32)
            .collect();
        let comp_cnt = Tensor::from_vec(cnt_v, s, &self.device)?; // [s]
                                                                  // Keep the first `cnt[t]` selected gids; sentinel the rest with `n_corpus`
                                                                  // (f32-exact since n_corpus ≤ 1024, and sorts after every valid gid), sort
                                                                  // the row ASCENDING (the compressed-index contract), then map the sentinel
                                                                  // to the `u32::MAX` skip-pad. Positions past `cnt[t]` are never read (the
                                                                  // kernel bounds iteration by `comp_cnt`) — the pad is for the contract.
        let colk = Tensor::arange(0u32, kmax as u32, &self.device)?.reshape((1, kmax))?;
        let keep = colk.broadcast_lt(&comp_cnt.reshape((s, 1))?)?; // [s, kmax] u8
        let sentinel = Tensor::full(n_corpus as u32, (s, kmax), &self.device)?;
        let masked = keep.where_cond(&order, &sentinel)?;
        let perm = masked.to_dtype(DType::F32)?.arg_sort_last_dim(true)?; // ascending
        let sorted = masked.gather(&perm, 1)?; // [s, kmax] ascending gids, sentinels last
        let maxpad = Tensor::full(u32::MAX, (s, kmax), &self.device)?;
        let comp_idx = sorted.lt(&sentinel)?.where_cond(&sorted, &maxpad)?; // sentinel → MAX
        Ok((comp_idx, comp_cnt, n_corpus))
    }

    /// Reference selector: the full Indexer top-k over ALL entries (no recall
    /// stage) — the oracle the two-stage path must reproduce. Host-evaluated
    /// (test/validation use only; the device argsort caps at 1024 columns).
    #[cfg(feature = "cuda")]
    pub fn full_indexer_top_k(
        &self,
        q_idx: &Tensor,
        weights: &Tensor,
        top_k: usize,
    ) -> Result<Vec<u32>> {
        if self.len == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        // Tier-aware: gather every key onto the GPU (a no-op when hot) so the
        // matmul runs against the query regardless of where the corpus lives.
        let all = Tensor::arange(0u32, self.len as u32, &self.device)?;
        let keys = self.gather_keys(&all)?;
        let scores = q_idx.matmul(&keys.t()?.contiguous()?)?.relu()?;
        let weighted = scores
            .broadcast_mul(&weights.reshape(((), 1))?)?
            .sum(0)?
            .to_vec1::<f32>()?;
        let mut order: Vec<u32> = (0..self.len as u32).collect();
        order.sort_by(|&a, &b| {
            weighted[b as usize]
                .partial_cmp(&weighted[a as usize])
                .unwrap()
        });
        let k = top_k.min(self.len);
        let mut ids = order[..k].to_vec();
        ids.sort_unstable();
        Ok(ids)
    }
}

/// A turn's compressed corpus in **native durable form** (Artifact C of
/// docs/deepseek_turn_seal_persistence.md): the two-region attended cache
/// (`nope_i8`/`nope_scale`/`rope_bf`) and the Indexer scoring `keys`, all
/// host-resident and self-describing. This is what the seal persists (no
/// re-quant — these already are the QAT storage precision) and what resume
/// injects back into a [`FloatGallery`].
///
/// **No positions are stored.** Group-start positions are a pure function of
/// entry order and the layer's compression ratio (`pos[i] = i · ratio`,
/// turn-relative), so they are *reconstructed* at inject time against the
/// position the turn lands at in the reconstructed context — exactly as the
/// chunked KV cache derives positions from cumulative layout rather than
/// persisting an absolute per token. Persisting the original absolute `pos`
/// would be a stale second source of truth that provenance re-layout
/// invalidates. The packed `signs` index is likewise rebuilt from `keys`
/// (`sign_pack`), and the archival f32 `attn` is not stored (reference
/// `gather_selected` only, never the resumed hot path).
#[derive(Clone, Debug, PartialEq)]
pub struct CorpusSnapshot {
    pub index_head_dim: usize,
    pub len: usize,
    pub nope_i8: Vec<u8>,     // len * NOPE_DIM
    pub nope_scale: Vec<f32>, // len * NOPE_BANDS
    pub rope_bf: Vec<u16>,    // len * ROPE_DIM  (bf16 bit patterns)
    pub keys: Vec<f32>,       // len * index_head_dim
}

impl CorpusSnapshot {
    const MAGIC: &'static [u8; 4] = b"DSC2";

    /// Serialize to a self-contained little-endian blob. Layout: magic(4) ·
    /// index_head_dim/len/nope_dim/nope_bands/rope_dim (5×u32) · nope_i8 bytes ·
    /// nope_scale f32 · rope_bf u16 · keys f32. Positions are NOT stored (they
    /// are reconstructed at inject time — see the type docs).
    pub fn encode(&self) -> Vec<u8> {
        let ih = self.index_head_dim;
        let len = self.len;
        let mut out =
            Vec::with_capacity(24 + len * (NOPE_DIM + NOPE_BANDS * 4 + ROPE_DIM * 2 + ih * 4));
        out.extend_from_slice(Self::MAGIC);
        for v in [
            ih as u32,
            len as u32,
            NOPE_DIM as u32,
            NOPE_BANDS as u32,
            ROPE_DIM as u32,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.nope_i8);
        for &v in &self.nope_scale {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.rope_bf {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.keys {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Self::encode`]. Returns `None` on foreign magic, geometry
    /// mismatch (a different single-latent shape), or a truncated payload.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 || &bytes[0..4] != Self::MAGIC {
            return None;
        }
        let rd_u32 =
            |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        let ih = rd_u32(4);
        let len = rd_u32(8);
        if rd_u32(12) != NOPE_DIM || rd_u32(16) != NOPE_BANDS || rd_u32(20) != ROPE_DIM {
            return None;
        }
        let mut off = 24;
        let take = |off: &mut usize, n: usize| -> Option<&[u8]> {
            let s = bytes.get(*off..*off + n)?;
            *off += n;
            Some(s)
        };
        let nope_i8 = take(&mut off, len * NOPE_DIM)?.to_vec();
        let nope_scale: Vec<f32> = take(&mut off, len * NOPE_BANDS * 4)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let rope_bf: Vec<u16> = take(&mut off, len * ROPE_DIM * 2)?
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let keys: Vec<f32> = take(&mut off, len * ih * 4)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some(Self {
            index_head_dim: ih,
            len,
            nope_i8,
            nope_scale,
            rope_bf,
            keys,
        })
    }
}

impl FloatGallery {
    /// Snapshot the live corpus into its native durable form
    /// ([`CorpusSnapshot`]) — the seal artifact. Reads the `len` live rows of
    /// the two-region cache + keys + pos back to host (tier-aware: the buffers
    /// may be on CPU once spilled). The signs index and archival `attn` are
    /// omitted (rebuilt / unused on restore).
    #[cfg(feature = "cuda")]
    pub fn snapshot(&self) -> Result<CorpusSnapshot> {
        let len = self.len;
        if len == 0 {
            return Ok(CorpusSnapshot {
                index_head_dim: self.index_head_dim,
                len: 0,
                nope_i8: Vec::new(),
                nope_scale: Vec::new(),
                rope_bf: Vec::new(),
                keys: Vec::new(),
            });
        }
        let nope_i8 = self
            .nope_i8
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let nope_scale = self
            .nope_scale
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rope_bf: Vec<u16> = self
            .rope_bf
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?
            .into_iter()
            .map(|b| b.to_bits())
            .collect();
        let keys = self
            .keys
            .narrow(0, 0, len)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(CorpusSnapshot {
            index_head_dim: self.index_head_dim,
            len,
            nope_i8,
            nope_scale,
            rope_bf,
            keys,
        })
    }

    /// Rebuild a gallery from a [`CorpusSnapshot`] — the resume injection. The
    /// two-region cache + keys are stamped directly (no re-quant); the sign
    /// index is repacked from `keys` (bit-identical to the live append, which
    /// packs the same keys); the archival `attn` is a CPU zero placeholder (the
    /// reference `gather_selected` is not used on the resumed hot path).
    ///
    /// `positions` (length `snap.len`) are the RECONSTRUCTED group-start
    /// positions the injected entries take in the resumed context — the caller
    /// computes them from where the turn lands in the reconstruction and the
    /// layer's compression ratio (`base + i·ratio`), NOT from any stored value.
    /// Given identical positions, the restored gallery is byte-identical to the
    /// live one on the durable set (`nope_i8`/`nope_scale`/`rope_bf`/`keys`/
    /// `pos`/`signs`).
    #[cfg(feature = "cuda")]
    pub fn from_snapshot(
        device: &Device,
        snap: &CorpusSnapshot,
        positions: &[u32],
    ) -> Result<Self> {
        let ih = snap.index_head_dim;
        if snap.len == 0 {
            return Self::new(device, NOPE_DIM + ROPE_DIM, ih, 1);
        }
        let len = snap.len;
        if positions.len() != len {
            candle::bail!(
                "from_snapshot: {} positions for {} entries",
                positions.len(),
                len
            );
        }
        let cap = len;
        let nope_i8 = Tensor::from_vec(snap.nope_i8.clone(), (len, NOPE_DIM), device)?;
        let nope_scale = Tensor::from_vec(snap.nope_scale.clone(), (len, NOPE_BANDS), device)?;
        let rope_bf = Tensor::from_vec(
            snap.rope_bf
                .iter()
                .map(|&b| half::bf16::from_bits(b))
                .collect::<Vec<_>>(),
            (len, ROPE_DIM),
            device,
        )?;
        let keys = Tensor::from_vec(snap.keys.clone(), (len, ih), device)?;
        let pos = Tensor::from_vec(positions.to_vec(), len, device)?;
        // Signs are a deterministic function of the keys — repack on the GPU
        // (identical to the live append's `sign_pack`).
        let signs = sign_pack(&keys)?;
        let mut g = Self {
            keys,
            nope_i8,
            nope_scale,
            rope_bf,
            signs,
            pos,
            len,
            cap,
            head_dim: NOPE_DIM + ROPE_DIM,
            index_head_dim: ih,
            device: device.clone(),
            spilled: false,
            region_ptr_cache: std::cell::Cell::new(None),
        };
        // Honor the hot/warm tier bound exactly as a live gallery would: past
        // HOT_ENTRY_CAP the float pair lives in CPU RAM (signs/pos stay GPU).
        g.maybe_spill(len)?;
        Ok(g)
    }
}

/// Build the two-region position-free cache (`latent_build_corpus_cache_kernel`)
/// for `n` rows of `attn_gpu` `[n, HEAD_DIM]` f32 into the gallery buffers at
/// `[row_lo, row_lo+n)`. The output views carry the row offset in their device
/// pointer, so the kernel (gid `0..n`, reading `attn_gpu[gid]`) writes exactly
/// those rows.
#[cfg(feature = "cuda")]
/// Append the raw bytes of `ids`' rows of a CPU-resident `[rows, w]` tensor to
/// `out` — the host-side row gather for the warm tier, straight into the packed
/// staging buffer so the re-heat never materialises a per-region tensor.
///
/// `live` bounds the gather to the tensor's occupied prefix, matching the
/// `narrow(0, 0, live)` the device path applies.
#[cfg(feature = "cuda")]
fn pack_rows(t: &Tensor, ids: &[u32], live: usize, row_bytes: usize, out: &mut Vec<u8>) -> Result<()> {
    use candle::{CpuStorage, Storage};
    let (storage, layout) = t.storage_and_layout();
    let cols = t.dim(1)?;
    let stride = layout.stride()[0];
    let base = layout.start_offset();
    // Row `id` of a CPU tensor as bytes, in the tensor's own element type.
    macro_rules! rows {
        ($v:expr, $to_bytes:expr) => {{
            let data = $v;
            for &id in ids {
                let id = id as usize;
                if id >= live {
                    candle::bail!("pack_rows: entry {id} past the live prefix ({live})");
                }
                let row = &data[base + id * stride..][..cols];
                for x in row {
                    out.extend_from_slice(&$to_bytes(*x));
                }
            }
        }};
    }
    let before = out.len();
    match &*storage {
        Storage::Cpu(CpuStorage::U8(v)) => {
            for &id in ids {
                let id = id as usize;
                if id >= live {
                    candle::bail!("pack_rows: entry {id} past the live prefix ({live})");
                }
                out.extend_from_slice(&v[base + id * stride..][..cols]);
            }
        }
        Storage::Cpu(CpuStorage::F32(v)) => rows!(v, |x: f32| x.to_ne_bytes()),
        Storage::Cpu(CpuStorage::BF16(v)) => rows!(v, |x: half::bf16| x.to_bits().to_ne_bytes()),
        Storage::Cpu(CpuStorage::U32(v)) => rows!(v, |x: u32| x.to_ne_bytes()),
        _ => candle::bail!(
            "pack_rows: unsupported warm-tier element type {:?}",
            t.dtype()
        ),
    }
    debug_assert_eq!(out.len() - before, ids.len() * row_bytes);
    Ok(())
}

/// One session's share of a wave's fleet-wide pooled block, for
/// [`append_batch_all`]: the gallery to append to, and which rows of the block
/// are its groups.
#[cfg(feature = "cuda")]
pub struct AppendSlice<'a> {
    pub gallery: &'a mut FloatGallery,
    /// `(first row of the block, rows)` runs, in the order they must land.
    ///
    /// Usually one. A SPECULATIVE VERIFY wave carries a whole drafted block as
    /// several consecutive decode rows of the SAME sequence, so that sequence
    /// occupies several wave slots and closes a group in more than one of them;
    /// every such run appends to this one gallery, back to back, exactly as the
    /// per-slot loop did.
    pub runs: Vec<(usize, usize)>,
}

#[cfg(feature = "cuda")]
impl AppendSlice<'_> {
    fn rows(&self) -> usize {
        self.runs.iter().map(|(_, n)| n).sum()
    }
}

/// Append EVERY session's completed groups in one pass over the wave.
///
/// A wave's compressor pool already emits one fleet-wide `[ΣG, d]` block with
/// each session's groups as a contiguous run of it, so the append is a scatter:
/// six arrays × one destination per session. Done per session that is six
/// `slice_set` launches times the wave's width times the compression layers,
/// each moving a few kilobytes — pure launch overhead, and exactly the
/// allocate-plus-copy-per-consumer shape hot-path invariant 2 forbids.
///
/// Here it is a fixed handful of launches for the WHOLE wave, independent of
/// session count: one sign-pack, one position upload, one latent seal, and one
/// [`rows_scatter`] whose descriptor table names every destination by address.
///
/// Each gallery's tier is settled BEFORE anything is written (`grow_to` then
/// `maybe_spill` over the post-append length), so every row is written once, to
/// its final home. That is the same bytes as building hot and spilling after —
/// a spill moves whole regions and does not care whether the new rows are in
/// them yet — and it removes the need to scatter twice around the crossing.
#[cfg(feature = "cuda")]
pub fn append_batch_all(
    slices: &mut [AppendSlice<'_>],
    attn_rows: &Tensor,
    key_rows: &Tensor,
    positions: &[u32],
    generation: &Generation,
) -> Result<()> {
    let (total, hd) = attn_rows.dims2()?;
    if total == 0 || slices.is_empty() {
        return Ok(());
    }
    // Every gallery of a wave shares one device; the seal and the scatter both
    // run there, so the incoming rows are placed on it once (a no-op when they
    // already are, which is the case for the wave's own pooled block).
    let device = slices[0].gallery.device.clone();
    let attn_rows = &attn_rows.to_device(&device)?;
    let key_rows = &key_rows.to_device(&device)?;
    let (kn, kd) = key_rows.dims2()?;
    if kn != total || positions.len() != total {
        candle::bail!(
            "append_batch_all: attn {:?}, keys {:?}, pos {} disagree on the row count",
            attn_rows.dims(),
            key_rows.dims(),
            positions.len()
        );
    }
    let covered: usize = slices.iter().map(|s| s.rows()).sum();
    if covered != total {
        candle::bail!("append_batch_all: slices cover {covered} rows of a {total}-row block");
    }
    for s in slices.iter() {
        for &(src_row, rows) in &s.runs {
            if src_row + rows > total {
                candle::bail!(
                    "append_batch_all: slice rows {src_row}..{} of a {total}-row block",
                    src_row + rows
                );
            }
        }
        if hd != s.gallery.head_dim || kd != s.gallery.index_head_dim {
            candle::bail!(
                "append_batch_all: attn {:?} / keys {:?} do not match the gallery \
                 (head_dim {}, index_head_dim {})",
                attn_rows.dims(),
                key_rows.dims(),
                s.gallery.head_dim,
                s.gallery.index_head_dim
            );
        }
    }

    // Sign-pack the new keys and upload their positions ONCE for the whole wave.
    // Both indexes are permanently GPU-resident (only the float pair spills), so
    // neither depends on any gallery's tier.
    let all_signs = sign_pack(key_rows)?;
    let all_pos = Tensor::from_vec(positions.to_vec(), total, &device)?;

    // ONE seal over the whole block. The two-region cache it builds IS the
    // stored latent — there is no separate f32 archive (the old canonical-f32
    // CPU copy was reference-only and its blocking D2H drained the pipeline on
    // every append). The kernel writes every row, so the scratch is
    // uninitialised (hot-path invariant 6).
    let all_nope = Tensor::empty((total, NOPE_DIM), DType::U8, &device)?;
    let all_scale = Tensor::empty((total, NOPE_BANDS), DType::F32, &device)?;
    let all_rope = Tensor::empty((total, ROPE_DIM), DType::BF16, &device)?;
    build_corpus_cache_into(attn_rows, &all_nope, &all_scale, &all_rope, 0, total)?;

    // Settle every gallery's capacity and tier before writing a single row.
    for s in slices.iter_mut() {
        let end = s.gallery.len + s.rows();
        s.gallery.grow_to(end)?;
        s.gallery.maybe_spill(end)?;
    }

    let mut runs: Vec<RowRun> = Vec::with_capacity(6 * slices.len());
    for s in slices.iter() {
        let g = &*s.gallery;
        let mut dst = g.len;
        for &(src_row, rows) in &s.runs {
            let take = |t: &Tensor| -> Result<Tensor> { t.narrow(0, src_row, rows) };
            runs.push(RowRun::new(take(&all_pos)?, &g.pos, dst));
            runs.push(RowRun::new(take(&all_signs)?, &g.signs, dst));
            if g.spilled {
                // Warm tier: the float pair lives in RAM, which the scatter
                // kernel cannot reach, so those four arrays take a
                // device-to-host copy each.
                let cpu = |t: &Tensor| -> Result<Tensor> { take(t)?.to_device(&Device::Cpu) };
                g.keys.slice_set(&cpu(key_rows)?, 0, dst)?;
                g.nope_i8.slice_set(&cpu(&all_nope)?, 0, dst)?;
                g.nope_scale.slice_set(&cpu(&all_scale)?, 0, dst)?;
                g.rope_bf.slice_set(&cpu(&all_rope)?, 0, dst)?;
            } else {
                runs.push(RowRun::new(take(key_rows)?, &g.keys, dst));
                runs.push(RowRun::new(take(&all_nope)?, &g.nope_i8, dst));
                runs.push(RowRun::new(take(&all_scale)?, &g.nope_scale, dst));
                runs.push(RowRun::new(take(&all_rope)?, &g.rope_bf, dst));
            }
            dst += rows;
        }
    }
    rows_scatter(&runs, generation)?;

    for s in slices.iter_mut() {
        s.gallery.len += s.rows();
    }
    Ok(())
}

fn build_corpus_cache_into(
    attn_gpu: &Tensor,
    nope_i8: &Tensor,
    nope_scale: &Tensor,
    rope_bf: &Tensor,
    row_lo: usize,
    n: usize,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    if n == 0 {
        return Ok(());
    }
    let dev = match attn_gpu.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("build_corpus_cache_into requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let ni8 = nope_i8.narrow(0, row_lo, n)?;
    let nsc = nope_scale.narrow(0, row_lo, n)?;
    let rbf = rope_bf.narrow(0, row_lo, n)?;
    macro_rules! p {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let (ptr, _g) = c.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                    ptr + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                }
                _ => candle::bail!("expected CUDA storage"),
            }
        }};
    }
    let comp_p = p!(attn_gpu, f32);
    let ni8_p = p!(&ni8, u8);
    let nsc_p = p!(&nsc, f32);
    let rbf_p = p!(&rbf, half::bf16);
    unsafe {
        candle_kernels::paged_latent::run_latent_build_corpus_cache(
            comp_p as *const f32,
            ni8_p as *mut u8,
            nsc_p as *mut f32,
            rbf_p as *mut core::ffi::c_void,
            0,
            n as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(())
}

/// Exact top-`m` entry ids by bounded u32 key (recall shortlist selector) —
/// valid at any entry count, fully on-device. Tie order is arbitrary: any
/// M-superset is a valid recall shortlist (the float rescore re-ranks it).
#[cfg(feature = "cuda")]
pub fn topm_select(counts: &Tensor, m: usize, bins: usize) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let g = counts.dim(0)?;
    let m = m.min(g);
    let dev = match counts.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("topm_select requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let hist = Tensor::zeros(bins, DType::U32, counts.device())?;
    let meta = Tensor::zeros(4, DType::U32, counts.device())?;
    let out = Tensor::zeros(m, DType::U32, counts.device())?;
    {
        let (sc, _) = counts.storage_and_layout();
        let (sh, _) = hist.storage_and_layout();
        let (sm, _) = meta.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        let (cp, _g1) = match &*sc {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (hp, _g2) = match &*sh {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (mp, _g3) = match &*sm {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (op, _g4) = match &*so {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let code = unsafe {
            candle_kernels::simple::bdp::run_topm_select(
                cp as *const u32,
                hp as *mut u32,
                mp as *mut u32,
                op as *mut u32,
                g as i32,
                m as i32,
                bins as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("deepseek_topm_select launch failed: cuda error {code}");
        }
    }
    Ok(out)
}

/// Pack sign bits of `[n, dim]` f32 rows into `[n, ceil(dim/32)]` u32 — the
/// on-device index the recall stage scans.
#[cfg(feature = "cuda")]
pub fn sign_pack(x: &Tensor) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let (n, dim) = x.dims2()?;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("sign_pack requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let x = x.contiguous()?;
    let out = Tensor::zeros((n, sign_words(dim)), DType::U32, x.device())?;
    {
        let (sx, _) = x.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        let (xp, _g1) = match &*sx {
            Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (op, _g2) = match &*so {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let code = unsafe {
            candle_kernels::simple::bdp::run_sign_pack(
                xp as *const f32,
                op as *mut u32,
                n as i32,
                dim as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("deepseek_sign_pack launch failed: cuda error {code}");
        }
    }
    Ok(out)
}

/// Sign-agreement counts `[g]` of every packed entry row against the packed
/// query heads (XNOR+popcount summed over heads and words).
#[cfg(feature = "cuda")]
pub fn bdp_recall(q_signs: &Tensor, signs: &Tensor, dim: usize) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let (h, w1) = q_signs.dims2()?;
    let (g, w2) = signs.dims2()?;
    if w1 != w2 || w1 != sign_words(dim) {
        candle::bail!("bdp_recall word mismatch: q {w1}, entries {w2}, dim {dim}");
    }
    let dev = match signs.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("bdp_recall requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let counts = Tensor::zeros(g, DType::U32, signs.device())?;
    {
        let (sq, _) = q_signs.storage_and_layout();
        let (ss, _) = signs.storage_and_layout();
        let (sc, _) = counts.storage_and_layout();
        let (qp, _g1) = match &*sq {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (sp, _g2) = match &*ss {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (cp, _g3) = match &*sc {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let code = unsafe {
            candle_kernels::simple::bdp::run_bdp_recall(
                qp as *const u32,
                sp as *const u32,
                cp as *mut u32,
                h as i32,
                g as i32,
                w1 as i32,
                dim as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("bdp_recall launch failed: cuda error {code}");
        }
    }
    Ok(counts)
}

/// Batched BDP recall across `n_sess` concurrent decode sessions in ONE launch.
/// `q_signs` `[n_sess·n_heads, words]` are the sessions' packed query heads (row
/// `s·n_heads + h`); `sign_tensors[s]` `[cnt[s], words]` is session `s`'s own
/// resident packed-sign index, **read in place** via a device pointer table (no
/// concatenation — that copy grew `O(Σlen·words)` with context depth). `off`/`cnt`
/// `[n_sess]` u32 give each session's base/count in the concatenated `counts`
/// output. Returns `counts` `[total_g]` — byte-identical per session to the
/// per-session [`bdp_recall`]. `max_g` = the largest `cnt[s]`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn bdp_recall_batched(
    q_signs: &Tensor,
    sign_tensors: &[Tensor],
    off: u64,
    cnt: u64,
    n_sess: usize,
    n_heads: usize,
    max_g: usize,
    dim: usize,
    generation: &Generation,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let words = sign_words(dim);
    let total_g: usize = sign_tensors
        .iter()
        .map(|t| t.dim(0))
        .sum::<Result<usize>>()?;
    let dev = match q_signs.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("bdp_recall_batched requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let counts = Tensor::zeros(total_g.max(1), DType::U32, q_signs.device())?;
    // Per-session base pointer table: each session's signs are read in place (the
    // sign index is always GPU-resident), so no depth-scaling `cat`.
    let mut sign_ptrs: Vec<i64> = Vec::with_capacity(n_sess);
    for t in sign_tensors {
        let (st, layout) = t.storage_and_layout();
        let p = match &*st {
            Storage::Cuda(c) => {
                c.as_cuda_slice::<u32>()?
                    .slice(layout.start_offset()..)
                    .device_ptr(&stream)
                    .0
            }
            _ => candle::bail!("bdp_recall_batched requires CUDA sign storage"),
        };
        sign_ptrs.push(p as i64);
    }
    // The per-gallery sign-pointer table rides the wave arena; it is read once
    // per block, which is what the bump arena is for.
    let sign_ptrs_t =
        crate::models::deepseek4::desc::stage(&sign_ptrs, generation)?;
    {
        let (sq, _) = q_signs.storage_and_layout();
        let (sc, _) = counts.storage_and_layout();
        let ptr_u32 = |st: &Storage| -> Result<u64> {
            match st {
                Storage::Cuda(c) => Ok(c.as_cuda_slice::<u32>()?.device_ptr(&stream).0),
                _ => unreachable!(),
            }
        };
        let table = sign_ptrs_t.ptr();
        let (qp, op, cp, outp) = (ptr_u32(&sq)?, off, cnt, ptr_u32(&sc)?);
        let code = unsafe {
            candle_kernels::simple::bdp::run_bdp_recall_batched(
                qp as *const u32,
                table as *const u64,
                op as *const u32,
                cp as *const u32,
                outp as *mut u32,
                n_sess as i32,
                n_heads as i32,
                max_g as i32,
                words as i32,
                dim as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("bdp_recall_batched launch failed: cuda error {code}");
        }
    }
    Ok(counts)
}

/// Batched exact top-`max_m` (per session `min(max_m, cnt[s])`) over per-session
/// count segments in ONE launch per stage. `counts` `[total_g]` concatenated,
/// `off`/`cnt` `[n_sess]` u32. Returns `[n_sess, max_m]` u32 of SESSION-RELATIVE
/// ids (`0..cnt[s]`); session `s`'s first `min(max_m, cnt[s])` columns are its
/// shortlist (remaining columns undefined). Byte-identical per session to
/// [`topm_select`]. `bins` = the agreement range (`n_heads·dim + 1`).
#[cfg(feature = "cuda")]
/// `off`/`cnt` are DEVICE ADDRESSES of `[n_sess]` u32 segment tables, not
/// tensors: the caller stages them once (from the wave arena where it has one)
/// and hands the same pair to this and to [`bdp_recall_batched`], instead of
/// each helper re-uploading its own copy.
#[allow(clippy::too_many_arguments)]
pub fn topm_select_batched(
    counts: &Tensor,
    off: u64,
    cnt: u64,
    n_sess: usize,
    max_g: usize,
    max_m: usize,
    bins: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let dev = match counts.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("topm_select_batched requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let hist = Tensor::zeros(n_sess * bins, DType::U32, counts.device())?;
    let meta = Tensor::zeros(n_sess * 4, DType::U32, counts.device())?;
    let out = Tensor::zeros((n_sess, max_m), DType::U32, counts.device())?;
    {
        let (sc, _) = counts.storage_and_layout();
        let (sh, _) = hist.storage_and_layout();
        let (sm, _) = meta.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        let ptr = |st: &Storage| -> Result<u64> {
            match st {
                Storage::Cuda(c) => Ok(c.as_cuda_slice::<u32>()?.device_ptr(&stream).0),
                _ => unreachable!(),
            }
        };
        let (cp, offp, cntp, hp, mp, outp) =
            (ptr(&sc)?, off, cnt, ptr(&sh)?, ptr(&sm)?, ptr(&so)?);
        let code = unsafe {
            candle_kernels::simple::bdp::run_topm_select_batched(
                cp as *const u32,
                offp as *const u32,
                cntp as *const u32,
                hp as *mut u32,
                mp as *mut u32,
                outp as *mut u32,
                n_sess as i32,
                max_g as i32,
                max_m as i32,
                bins as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("topm_select_batched launch failed: cuda error {code}");
        }
    }
    Ok(out)
}

/// Batched two-stage selection across concurrent decode sessions. Runs Stage 1
/// (BDP recall + top-M shortlist) for the WHOLE wave in one launch per kernel —
/// the per-session loop's dominant cost (`topm_select` = a single-warp serial
/// bin scan × sessions) collapses to a single grid — then does the bounded
/// Stage-2 float rescore per session (a few-column cuBLAS matmul + argsort over
/// ≤`top_m` keys). Each session's gallery is independent and, at decode, wholly
/// causal, so this is byte-identical per session to calling
/// [`FloatGallery::two_stage_select`] in a loop — the shortlist is the same
/// M-superset and the float rescore picks the same top-k. Returns per session
/// `(gids_ascending [k] u32, k)`; empty/degenerate sessions get `(zeros(1), 0)`.
#[cfg(feature = "cuda")]
pub fn two_stage_select_batched(
    galleries: &[&FloatGallery],
    queries: &[Tensor],
    weights: &[Tensor],
    top_m: usize,
    top_k: usize,
    generation: &Generation,
) -> Result<Vec<(Tensor, usize)>> {
    let n_sess = galleries.len();
    if n_sess == 0 {
        return Ok(Vec::new());
    }
    let dev = galleries[0].device.clone();
    let ihd = galleries[0].index_head_dim;
    let n_heads = queries[0].dim(0)?;
    let empty = || -> Result<(Tensor, usize)> { Ok((Tensor::zeros(1, DType::U32, &dev)?, 0)) };
    let max_m = top_m.clamp(1, 1024);

    // ── Partition: only DEEP sessions (`len > max_m`) need the recall stage.
    // Shallow sessions (`0 < len ≤ max_m`) take the whole gallery as the
    // shortlist — the per-session path's all-pass fast branch — so a batch of
    // short conversations pays no Stage-1 cost at all (the batched recall win is
    // realized exactly where it exists: unbounded-depth galleries). ──
    let deep_idx: Vec<usize> = (0..n_sess).filter(|&s| galleries[s].len > max_m).collect();

    // ── Stage 1 (batched) over the deep sessions only ──
    // `shortlist_deep[d]` (row d = deep_idx[d]) holds session-relative top-M ids.
    let shortlist_deep: Option<Tensor> = if deep_idx.is_empty() {
        None
    } else {
        let nd = deep_idx.len();
        let q_signs_cat = Tensor::cat(
            &deep_idx
                .iter()
                .map(|&s| sign_pack(&queries[s]))
                .collect::<Result<Vec<_>>>()?,
            0,
        )?; // [nd*h, words]
            // Per-session resident sign indices, read in place via a device pointer table
            // inside `bdp_recall_batched` — no depth-scaling concatenation. The gallery owns
            // each `signs` buffer (never spilled), so these views stay valid for the launch.
        let mut sign_tensors: Vec<Tensor> = Vec::with_capacity(nd);
        let mut off = Vec::with_capacity(nd);
        let mut cnt = Vec::with_capacity(nd);
        let mut running = 0u32;
        let mut max_g = 0usize;
        for &s in &deep_idx {
            let len = galleries[s].len;
            off.push(running);
            cnt.push(len as u32);
            sign_tensors.push(galleries[s].packed_signs()?);
            running += len as u32;
            max_g = max_g.max(len);
        }
        // Staged ONCE and shared by the recall and the top-M below, rather than
        // each helper uploading its own copy of the same segment table.
        let off_t = crate::models::deepseek4::desc::stage_slice(&off, generation)?;
        let cnt_t = crate::models::deepseek4::desc::stage_slice(&cnt, generation)?;
        let bins = n_heads * ihd + 1;
        let counts = bdp_recall_batched(
            &q_signs_cat,
            &sign_tensors,
            off_t.ptr(),
            cnt_t.ptr(),
            nd,
            n_heads,
            max_g,
            ihd,
            generation,
        )?;
        Some(topm_select_batched(
            &counts,
            off_t.ptr(),
            cnt_t.ptr(),
            nd,
            max_g,
            max_m,
            bins,
        )?) // [nd, max_m]
    };

    // ── Stage 2 (batched): float rescore over the per-session shortlists ──
    // Phase A gathers each session's ≤`max_m` shortlist keys (tier-aware — from
    // CPU RAM when the gallery is spilled, in place otherwise). Phase B then
    // rescores the WHOLE batch in one padded `bmm` + one batched argsort + one
    // batched gather, replacing the per-session matmul/argsort launch loop. The
    // padding columns (`j ≥ mₛ`) are masked to −∞ so they never enter a top-k.
    // The selected gids are returned in descending-score order, not ascending:
    // the decode reader gathers them into a dense `comp_idx` block and attends
    // the SET (softmax is order-independent; each entry carries its own position
    // for RoPE-at-read), so order is immaterial — and the byte-for-byte per
    // session equivalence the tests hold is over the selected gid SET.
    let mut out_slots: Vec<Option<(Tensor, usize)>> = (0..n_sess).map(|_| None).collect();
    // Phase A — per-session shortlist + tier-aware key gather.
    struct Active {
        s: usize,
        sl: Tensor,   // [m_s] session-relative shortlist ids
        keys: Tensor, // [m_s, ihd] gathered scoring keys (GPU)
        m_s: usize,
        k: usize,
    }
    let mut active: Vec<Active> = Vec::new();
    let mut deep_cursor = 0usize;
    for (s, (g, _q)) in galleries.iter().zip(queries).enumerate() {
        let len = g.len;
        if len == 0 || top_k == 0 {
            out_slots[s] = Some(empty()?);
            continue;
        }
        let m_s = max_m.min(len);
        let sl = if len > max_m {
            let row = shortlist_deep
                .as_ref()
                .expect("deep sessions produced a shortlist")
                .narrow(0, deep_cursor, 1)?
                .reshape(max_m)?
                .narrow(0, 0, m_s)?;
            deep_cursor += 1;
            row
        } else {
            Tensor::arange(0u32, len as u32, &dev)?
        };
        let keys = g.gather_keys(&sl)?; // [m_s, ihd]
        active.push(Active {
            s,
            sl,
            keys,
            m_s,
            k: top_k.min(m_s),
        });
    }
    if active.is_empty() {
        return Ok(out_slots.into_iter().map(|o| o.expect("filled")).collect());
    }

    // Phase B — pad to the batch's widest shortlist and rescore all at once.
    let big = active.len();
    let mm = active.iter().map(|a| a.m_s).max().unwrap();
    let max_k = active.iter().map(|a| a.k).max().unwrap();
    let keys_pad: Vec<Tensor> = active
        .iter()
        .map(|a| {
            if a.m_s == mm {
                Ok(a.keys.clone())
            } else {
                let pad = Tensor::zeros((mm - a.m_s, ihd), DType::F32, &dev)?;
                Tensor::cat(&[&a.keys, &pad], 0)
            }
        })
        .collect::<Result<_>>()?;
    let keys_all = Tensor::stack(&keys_pad, 0)?; // [big, mm, ihd]
    let q_all = Tensor::stack(
        &active
            .iter()
            .map(|a| queries[a.s].clone())
            .collect::<Vec<_>>(),
        0,
    )?; // [big, h, ihd]
    let w_all = Tensor::stack(
        &active
            .iter()
            .map(|a| weights[a.s].clone())
            .collect::<Vec<_>>(),
        0,
    )?; // [big, h]
        // scores[b,h,j] = q·k ; weighted[b,j] = Σ_h relu(scores)·w
    // The `contiguous()` on the transpose is a known invariant-2 violation, kept
    // for now because deleting it is NOT yet justified by measurement. candle's
    // matmul does accept the strided transpose — every gate stays green without
    // the copy, here and at the three sibling sites — but the model-level A/B
    // came back inside run-to-run variance (cfg8 bulk 790.5 without vs 802.8
    // with, against a 820.4 reading from an identical earlier build; cfg16
    // decode 54.2 vs 54.7 vs 56.6). A copy this size should show up, and it does
    // not, which points at cuBLAS picking a slower kernel for the strided
    // operand and roughly cancelling the saved traffic. Removing it is still the
    // right end state; it needs the GEMM work to land first, not just the
    // deletion, and it needs an nsys attribution of the ucopy_f32 population
    // rather than a whole-model sweep that cannot resolve it.
    let scores = q_all.matmul(&keys_all.transpose(1, 2)?.contiguous()?)?; // [big,h,mm]
    // relu, weight, sum over heads and the padding mask, in ONE launch. Padding
    // keys score 0, which could outrank a genuinely negative-scoring real entry,
    // so masked columns take −1e30 and stay out of the top-k. Only the tiny
    // [big] count vector is uploaded, never a [big·mm] host-built mask.
    let counts: Vec<u32> = active.iter().map(|a| a.m_s as u32).collect();
    let counts = Tensor::from_vec(counts, big, &dev)?;
    let weighted = indexer_score_reduce(&scores, &w_all, Some(&counts))?;
    // One batched argsort (descending) + one batched gather of the top-max_k
    // shortlist-relative ids through each session's `sl` → absolute entry ids.
    let order = weighted
        .arg_sort_last_dim(false)?
        .narrow(1, 0, max_k)?
        .contiguous()?; // [big,max_k]
    let sl_pad: Vec<Tensor> = active
        .iter()
        .map(|a| {
            if a.m_s == mm {
                Ok(a.sl.clone())
            } else {
                let pad = Tensor::zeros(mm - a.m_s, DType::U32, &dev)?;
                Tensor::cat(&[&a.sl, &pad], 0)
            }
        })
        .collect::<Result<_>>()?;
    let sl_all = Tensor::stack(&sl_pad, 0)?; // [big, mm]
    let gids_all = sl_all.gather(&order, 1)?; // [big, max_k] absolute ids, score order

    // Phase C — hand back each session's first `k` (varies with shortlist size).
    for (i, a) in active.iter().enumerate() {
        let gids = gids_all
            .narrow(0, i, 1)?
            .reshape(max_k)?
            .narrow(0, 0, a.k)?
            .contiguous()?;
        out_slots[a.s] = Some((gids, a.k));
    }
    Ok(out_slots.into_iter().map(|o| o.expect("filled")).collect())
}

/// Gather EVERY decode session's selected corpus rows into a shared pre-allocated
/// block in ONE kernel launch (across all HOT galleries), instead of a launch per
/// session. `gids[i]` are session `i`'s selected entry ids, placed at output row
/// `row_offsets[i]`. Hot galleries batch through one launch via a device pointer
/// table (each gallery's region base addresses); a spilled gallery re-heats its
/// rows on CPU and `slice_set`s them in (the bounded warm-tier path). Byte-
/// identical per row to [`FloatGallery::gather_corpus_into`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gather_corpus_batched(
    galleries: &[&FloatGallery],
    gids: &[Tensor],
    row_offsets: &[u32],
    out_nope: &Tensor,
    out_scale: &Tensor,
    out_rope: &Tensor,
    out_pos: &Tensor,
    generation: &Generation,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let n = galleries.len();
    if n == 0 {
        return Ok(());
    }
    // Spilled sessions take the per-session CPU path; collect the hot ones.
    let mut hot: Vec<usize> = Vec::new();
    for i in 0..n {
        if gids[i].dim(0)? == 0 {
            continue;
        }
        let g = galleries[i];
        if g.spilled {
            g.gather_corpus_into(
                &gids[i],
                out_nope,
                out_scale,
                out_rope,
                out_pos,
                row_offsets[i] as usize,
                generation,
            )?;
        } else {
            hot.push(i);
        }
    }
    if hot.is_empty() {
        return Ok(());
    }
    let device = galleries[hot[0]].device.clone();
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("gather_corpus_batched requires CUDA"),
    };
    let stream = dev.cuda_stream();
    macro_rules! dp {
        ($t:expr, $ty:ty) => {{
            let (s, _) = $t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?.device_ptr(&stream).0,
                _ => candle::bail!("gather_corpus_batched: non-CUDA storage"),
            }
        }};
    }

    // Packed metadata (exactly 2 small uploads): `ptrs` = the device pointer
    // table [nope|scale|rope|pos|gids] (region addresses cached per gallery; the
    // gid address is this step's selection tensor), `meta` = [out_off|cnt]. No
    // per-session `narrow`/lock for regions, no gid `cat`, no per-array upload.
    let n_hot = hot.len();
    let mut ptrs = vec![0i64; 5 * n_hot];
    let mut meta = vec![0u32; 2 * n_hot];
    let mut max_k = 0usize;
    // Keep the (possibly freshly-made-contiguous) gid tensors alive until the
    // launch is enqueued — their addresses are in the pointer table.
    let mut gid_keep: Vec<Tensor> = Vec::with_capacity(n_hot);
    for (col, &i) in hot.iter().enumerate() {
        let g = galleries[i];
        let rp = g.hot_region_ptrs()?;
        ptrs[col] = rp[0] as i64;
        ptrs[n_hot + col] = rp[1] as i64;
        ptrs[2 * n_hot + col] = rp[2] as i64;
        ptrs[3 * n_hot + col] = rp[3] as i64;
        let gk = gids[i].contiguous()?;
        ptrs[4 * n_hot + col] = dp!(gk, u32) as i64;
        meta[col] = row_offsets[i]; // out_off
        meta[n_hot + col] = gk.dim(0)? as u32; // cnt
        max_k = max_k.max(gk.dim(0)?);
        gid_keep.push(gk);
    }
    // Both tables ride the wave arena — they are read once per block and never
    // touched again, which is exactly what the bump arena is for. Uploading them
    // per call cost more than this kernel does (see `desc.rs`).
    let ptrs_t = crate::models::deepseek4::desc::stage(&ptrs, generation)?;
    let meta_t = crate::models::deepseek4::desc::stage_slice(&meta, generation)?;

    let code = unsafe {
        candle_kernels::simple::corpus_gather::run_corpus_gather_rows_batched(
            ptrs_t.ptr() as *const i64,
            meta_t.ptr() as *const u32,
            dp!(out_nope, u8) as *mut u32,
            dp!(out_scale, f32) as *mut u32,
            dp!(out_rope, half::bf16) as *mut u32,
            dp!(out_pos, u32) as *mut u32,
            n_hot as i32,
            max_k as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if code != 0 {
        candle::bail!("corpus_gather_rows_batched launch failed: cuda error {code}");
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::IndexOp;

    const HD: usize = 512;
    const IH: usize = 128;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Full 32-bit draw: `>> 33` would keep only 31 bits and map every
        // value into [-1, 0) — all-negative data zeroes every sign bit and
        // silently blinds the sign-agreement recall.
        (((*seed >> 32) as u32 as f64) / (u32::MAX as f64) * 2.0 - 1.0) as f32
    }

    fn rows(n: usize, d: usize, seed: &mut u64) -> Vec<f32> {
        (0..n * d).map(|_| lcg(seed)).collect()
    }

    /// The fused `gather_corpus_into` (one kernel launch per session, writing
    /// straight into a shared block at a row offset) must be BIT-IDENTICAL to the
    /// per-region `index_select` path (`gather_corpus`), for every one of the
    /// four hot-cache regions and at a non-zero destination offset.
    #[test]
    #[ignore]
    fn gather_corpus_into_matches_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut seed = 0xC0FFEEu64;
        let n = 200usize;
        let mut g = FloatGallery::new(&dev, HD, IH, 128)?;
        let attn = Tensor::from_vec(rows(n, HD, &mut seed), (n, HD), &dev)?;
        let keys = Tensor::from_vec(rows(n, IH, &mut seed), (n, IH), &dev)?;
        let positions: Vec<u32> = (0..n as u32).map(|i| i * 4).collect();
        g.append_batch(&attn, &keys, &positions)?;

        let gid_v: Vec<u32> = vec![5, 17, 42, 3, 199, 0, 128, 63];
        let k = gid_v.len();
        let gids = Tensor::from_vec(gid_v, k, &dev)?;

        // Reference: the per-region gather.
        let (rn, rs, rr, rp) = g.gather_corpus(&gids)?;

        // Fused: gather into a shared block at offset 2 (test the placement).
        let off = 2usize;
        let total = k + off + 1; // trailing untouched row
        let on = Tensor::zeros((total, NOPE_DIM), DType::U8, &dev)?;
        let os = Tensor::zeros((total, NOPE_BANDS), DType::F32, &dev)?;
        let or_ = Tensor::zeros((total, ROPE_DIM), DType::BF16, &dev)?;
        let op = Tensor::zeros(total, DType::U32, &dev)?;
        let scope = crate::models::deepseek4::desc::scope(&dev)?;
        g.gather_corpus_into(&gids, &on, &os, &or_, &op, off, &scope)?;

        let got_n = on.narrow(0, off, k)?;
        let got_s = os.narrow(0, off, k)?;
        let got_r = or_.narrow(0, off, k)?;
        let got_p = op.narrow(0, off, k)?;

        let bits_u8 = |t: &Tensor| -> Result<Vec<u8>> { t.flatten_all()?.to_vec1::<u8>() };
        let bits_u32 = |t: &Tensor| -> Result<Vec<u32>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        assert_eq!(bits_u8(&got_n)?, bits_u8(&rn)?, "nope_i8 mismatch");
        assert_eq!(bits_u32(&got_s)?, bits_u32(&rs)?, "nope_scale mismatch");
        assert_eq!(bits_u32(&got_r)?, bits_u32(&rr)?, "rope_bf mismatch");
        assert_eq!(
            got_p.to_vec1::<u32>()?,
            rp.to_vec1::<u32>()?,
            "pos mismatch"
        );
        Ok(())
    }

    /// The single-launch `gather_corpus_batched` across several galleries must be
    /// BIT-IDENTICAL, per session, to the per-session `gather_corpus` — the
    /// device pointer table must address each gallery's rows correctly and each
    /// session's block land at its own output offset.
    #[test]
    #[ignore]
    fn gather_corpus_batched_matches_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut seed = 0xBA7C4EDu64;
        let ks = [7usize, 3, 12, 1, 9];
        let mut galleries: Vec<FloatGallery> = Vec::new();
        let mut gids: Vec<Tensor> = Vec::new();
        let mut refs: Vec<(Tensor, Tensor, Tensor, Tensor)> = Vec::new();
        let mut offsets: Vec<u32> = Vec::new();
        let mut cum = 0u32;
        for (s, &k) in ks.iter().enumerate() {
            let entries = 40 + s * 30;
            let mut g = FloatGallery::new(&dev, HD, IH, 64)?;
            let attn = Tensor::from_vec(rows(entries, HD, &mut seed), (entries, HD), &dev)?;
            let keys = Tensor::from_vec(rows(entries, IH, &mut seed), (entries, IH), &dev)?;
            let positions: Vec<u32> = (0..entries as u32).map(|i| i * 4).collect();
            g.append_batch(&attn, &keys, &positions)?;
            let gv: Vec<u32> = (0..k).map(|j| ((j * 3 + s) % entries) as u32).collect();
            let gt = Tensor::from_vec(gv, k, &dev)?;
            refs.push(g.gather_corpus(&gt)?);
            gids.push(gt);
            offsets.push(cum);
            cum += k as u32;
            galleries.push(g);
        }
        let total = cum as usize;
        let on = Tensor::zeros((total, NOPE_DIM), DType::U8, &dev)?;
        let os = Tensor::zeros((total, NOPE_BANDS), DType::F32, &dev)?;
        let or_ = Tensor::zeros((total, ROPE_DIM), DType::BF16, &dev)?;
        let op = Tensor::zeros(total, DType::U32, &dev)?;
        let grefs: Vec<&FloatGallery> = galleries.iter().collect();
        let scope = crate::models::deepseek4::desc::scope(&dev)?;
        gather_corpus_batched(&grefs, &gids, &offsets, &on, &os, &or_, &op, &scope)?;

        let bits_u8 = |t: &Tensor| -> Result<Vec<u8>> { t.flatten_all()?.to_vec1::<u8>() };
        let bits_u32 = |t: &Tensor| -> Result<Vec<u32>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        for (s, &k) in ks.iter().enumerate() {
            let o = offsets[s] as usize;
            let (rn, rs, rr, rp) = &refs[s];
            assert_eq!(
                bits_u8(&on.narrow(0, o, k)?)?,
                bits_u8(rn)?,
                "sess {s} nope"
            );
            assert_eq!(
                bits_u32(&os.narrow(0, o, k)?)?,
                bits_u32(rs)?,
                "sess {s} scale"
            );
            assert_eq!(
                bits_u32(&or_.narrow(0, o, k)?)?,
                bits_u32(rr)?,
                "sess {s} rope"
            );
            assert_eq!(
                op.narrow(0, o, k)?.to_vec1::<u32>()?,
                rp.to_vec1::<u32>()?,
                "sess {s} pos"
            );
        }
        Ok(())
    }

    /// Diagnostic: candle CUDA argsort sanity across widths (stream + width
    /// limits) — the two-stage selector leans on it.
    #[test]
    #[ignore]
    fn argsort_cuda_probe() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for n in [64usize, 500, 1024, 1500, 2000, 4096] {
            let vals: Vec<f32> = (0..n).map(|i| ((i * 7919) % n) as f32).collect();
            let t = Tensor::from_vec(vals.clone(), (1, n), &dev)?;
            let order = t.arg_sort_last_dim(false)?.squeeze(0)?.to_vec1::<u32>()?;
            let max = order.iter().max().copied().unwrap_or(0);
            let uniq: std::collections::HashSet<_> = order.iter().collect();
            eprintln!(
                "[argsort] n={n}: max_idx={max} unique={} first4={:?}",
                uniq.len(),
                &order[..4]
            );
            if n <= 1024 {
                assert!(
                    (max as usize) < n && uniq.len() == n,
                    "argsort broken at n={n} (within its supported width)"
                );
            }
            // n > 1024: known-broken (single-block bitonic) — printed above for
            // visibility; the gallery never relies on it past 1024.
        }
        Ok(())
    }

    /// The recall kernel in isolation: hand-uploaded bit patterns vs a host
    /// XNOR+popcount reference.
    #[test]
    #[ignore]
    fn bdp_recall_matches_host() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 21u64;
        let (h, g, words, dim) = (4usize, 33usize, 4usize, 128usize);
        let qs: Vec<u32> = (0..h * words)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 24) as u32
            })
            .collect();
        let es: Vec<u32> = (0..g * words)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 24) as u32
            })
            .collect();
        let q_t = Tensor::from_vec(qs.clone(), (h, words), &dev)?;
        let e_t = Tensor::from_vec(es.clone(), (g, words), &dev)?;
        let counts = bdp_recall(&q_t, &e_t, dim)?.to_vec1::<u32>()?;
        for e in 0..g {
            let mut expect = 0u32;
            for hh in 0..h {
                for w in 0..words {
                    expect += (!(qs[hh * words + w] ^ es[e * words + w])).count_ones();
                }
            }
            assert_eq!(counts[e], expect, "entry {e}: {} vs {expect}", counts[e]);
        }
        Ok(())
    }

    /// Bisection: does prior gallery activity (allocations, slice_set,
    /// growth, candle matmul) corrupt a subsequent direct sign_pack?
    #[test]
    #[ignore]
    fn pack_after_gallery_activity() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 41u64;
        let check = |label: &str, dev: &Device, s: &mut u64| -> Result<()> {
            let qv = rows(2, IH, s);
            let t = Tensor::from_vec(qv.clone(), (2, IH), dev)?;
            let packed = sign_pack(&t)?.to_vec2::<u32>()?;
            let mut expect0 = [0u32; 4];
            for d in 0..IH {
                if qv[d] >= 0.0 {
                    expect0[d / 32] |= 1 << (d % 32);
                }
            }
            assert_eq!(
                packed[0],
                expect0.to_vec(),
                "sign_pack broken after: {label}"
            );
            Ok(())
        };
        check("nothing", &dev, &mut s)?;

        let mut gal = FloatGallery::new(&dev, HD, IH, 4)?;
        check("gallery alloc", &dev, &mut s)?;

        let attn = rows(3, HD, &mut s);
        let keys = rows(3, IH, &mut s);
        gal.append_batch(
            &Tensor::from_vec(attn, (3, HD), &dev)?,
            &Tensor::from_vec(keys, (3, IH), &dev)?,
            &[0, 4, 8],
        )?;
        check("first append", &dev, &mut s)?;

        // Force growth (4 → 16).
        let attn = rows(9, HD, &mut s);
        let keys = rows(9, IH, &mut s);
        gal.append_batch(
            &Tensor::from_vec(attn, (9, HD), &dev)?,
            &Tensor::from_vec(keys, (9, IH), &dev)?,
            &[12, 16, 20, 24, 28, 32, 36, 40, 44],
        )?;
        check("growth append", &dev, &mut s)?;

        let q = Tensor::from_vec(rows(4, IH, &mut s), (4, IH), &dev)?;
        let w = Tensor::from_vec(vec![1.0f32; 4], 4, &dev)?;
        let _ = gal.full_indexer_top_k(&q, &w, 4)?;
        check("full_indexer (matmul)", &dev, &mut s)?;

        // The failing scale: one big append (500 rows → growth to 512).
        let n = 500usize;
        let attn = rows(n, HD, &mut s);
        let keys = rows(n, IH, &mut s);
        let pos: Vec<u32> = (0..n as u32).map(|i| 48 + i * 4).collect();
        gal.append_batch(
            &Tensor::from_vec(attn, (n, HD), &dev)?,
            &Tensor::from_vec(keys.clone(), (n, IH), &dev)?,
            &pos,
        )?;
        check("big append (500 rows)", &dev, &mut s)?;
        let _ = gal.full_indexer_top_k(&q, &w, 8)?;
        check("full_indexer big", &dev, &mut s)?;

        // And the gallery's own stored signs at scale: entry 100's packed
        // signs must match the host packing of what was appended.
        let got = gal.packed_signs()?.i(12 + 100)?.to_vec1::<u32>()?;
        let mut expect = [0u32; 4];
        for d in 0..IH {
            if keys[100 * IH + d] >= 0.0 {
                expect[d / 32] |= 1 << (d % 32);
            }
        }
        assert_eq!(got, expect.to_vec(), "stored signs at scale");

        // Pack the SAME tensor that already flowed through the matmul (the
        // failing tests reuse `q`; fresh-tensor checks can't see this).
        let qv_host = q.flatten_all()?.to_vec1::<f32>()?;
        let packed_q = sign_pack(&q)?.to_vec2::<u32>()?;
        let mut expect_q0 = [0u32; 4];
        for d in 0..IH {
            if qv_host[d] >= 0.0 {
                expect_q0[d / 32] |= 1 << (d % 32);
            }
        }
        assert_eq!(packed_q[0], expect_q0.to_vec(), "sign_pack of the reused q");
        Ok(())
    }

    /// The pack→recall CHAIN (kernel output feeding the next kernel with no
    /// host read-back in between) vs host reference.
    #[test]
    #[ignore]
    fn pack_recall_chain_matches_host() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 31u64;
        let (h, g) = (4usize, 55usize);
        let qv = rows(h, IH, &mut s);
        let ev = rows(g, IH, &mut s);
        let q_t = Tensor::from_vec(qv.clone(), (h, IH), &dev)?;
        let e_t = Tensor::from_vec(ev.clone(), (g, IH), &dev)?;
        let q_signs = sign_pack(&q_t)?;
        let e_signs = sign_pack(&e_t)?;
        let counts = bdp_recall(&q_signs, &e_signs, IH)?.to_vec1::<u32>()?;
        let pack_host = |v: &[f32], r: usize| -> [u32; 4] {
            let mut out = [0u32; 4];
            for d in 0..IH {
                if v[r * IH + d] >= 0.0 {
                    out[d / 32] |= 1 << (d % 32);
                }
            }
            out
        };
        for e in 0..g {
            let es = pack_host(&ev, e);
            let mut expect = 0u32;
            for hh in 0..h {
                let qs = pack_host(&qv, hh);
                for w in 0..4 {
                    expect += (!(qs[w] ^ es[w])).count_ones();
                }
            }
            assert_eq!(counts[e], expect, "entry {e}");
        }
        Ok(())
    }

    /// Gate (b): `keys`/`pos` read back bit-for-bit across the grow-triggering
    /// second append (lossless tiers), and `attn_entries` reconstructs the dense
    /// f32 EXACTLY as the deterministic two-region dequant of the input (int8
    /// nope · per-band amax ‖ bf16 rope) — the f32 archive is gone, so the
    /// reconstruction equals what the kernels read, asserted against the
    /// recomputed quantized bytes (no tolerance).
    #[test]
    #[ignore]
    fn gallery_round_trip_raw_bytes() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut g = FloatGallery::new(&dev, HD, IH, 4)?;
        let mut s = 3u64;
        // Two appends that force a growth (4 → 16 cap).
        for (n, base) in [(3usize, 0u32), (9, 12)] {
            let attn = rows(n, HD, &mut s);
            let keys = rows(n, IH, &mut s);
            let pos: Vec<u32> = (0..n as u32).map(|i| base + i * 4).collect();
            let attn_t = Tensor::from_vec(attn.clone(), (n, HD), &dev)?;
            let keys_t = Tensor::from_vec(keys.clone(), (n, IH), &dev)?;
            let before = g.len();
            g.append_batch(&attn_t, &keys_t, &pos)?;
            assert_eq!(g.len(), before + n);
            let got_attn = g
                .attn_entries()?
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let got_keys = g
                .scoring_keys()?
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            // `attn_entries` reconstructs the dense f32 from the lossy two-region,
            // so it is NOT the input. Its contract is to be the EXACT dequant of
            // the gallery's OWN stored bytes: nope band `= int8 · per-band scale`,
            // rope tail `= bf16 → f32`. Recompute that from the stored tensors and
            // assert bit-for-bit (the accessor reads the right rows post-growth and
            // applies the scale correctly). The rope tail is additionally a true
            // bf16 passthrough of the input. The input→int8 quantization itself is
            // gated bit-exact by the `build_corpus_cache` mirror test.
            let ni8: Vec<u8> = g
                .nope_i8
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<u8>()?;
            let nsc: Vec<f32> = g
                .nope_scale
                .i((before..before + n, ..))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for i in 0..n {
                for band in 0..NOPE_BANDS {
                    let scale = nsc[i * NOPE_BANDS + band];
                    for j in 0..SUB_DIM {
                        let d = band * SUB_DIM + j;
                        let expected = (ni8[i * NOPE_DIM + d] as i8 as f32) * scale;
                        assert_eq!(
                            got_attn[i * HD + d].to_bits(),
                            expected.to_bits(),
                            "nope attn[{i}][{d}]"
                        );
                    }
                }
                for d in NOPE_DIM..HD {
                    let expected = half::bf16::from_f32(attn[i * HD + d]).to_f32();
                    assert_eq!(
                        got_attn[i * HD + d].to_bits(),
                        expected.to_bits(),
                        "rope attn[{i}][{d}] not a bf16 passthrough of the input"
                    );
                }
            }
            for (i, (&a, &b)) in keys.iter().zip(&got_keys).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "keys[{i}]");
            }
            let got_pos = g.positions()?.to_vec1::<u32>()?;
            assert_eq!(&got_pos[before..before + n], &pos[..]);
        }
        Ok(())
    }

    /// `CorpusSnapshot` byte codec: encode → decode is the identity, and the
    /// header bytes are exactly the documented little-endian layout (magic +
    /// 5×u32 geometry). Pure host logic — asserts raw bytes, not tolerances.
    #[test]
    fn corpus_snapshot_codec_round_trip() -> Result<()> {
        let ih = IH; // 128
        let len = 3usize;
        let snap = CorpusSnapshot {
            index_head_dim: ih,
            len,
            nope_i8: (0..len * NOPE_DIM).map(|i| (i % 251) as u8).collect(),
            nope_scale: (0..len * NOPE_BANDS)
                .map(|i| i as f32 * 0.5 - 3.0)
                .collect(),
            rope_bf: (0..len * ROPE_DIM).map(|i| (i * 7 + 1) as u16).collect(),
            keys: (0..len * ih).map(|i| (i as f32).sin()).collect(),
        };
        let bytes = snap.encode();

        // Exact header: "DSC2" · ih · len · NOPE_DIM · NOPE_BANDS · ROPE_DIM.
        assert_eq!(&bytes[0..4], b"DSC2");
        assert_eq!(&bytes[4..8], &(ih as u32).to_le_bytes());
        assert_eq!(&bytes[8..12], &(len as u32).to_le_bytes());
        assert_eq!(&bytes[12..16], &(NOPE_DIM as u32).to_le_bytes());
        assert_eq!(&bytes[16..20], &(NOPE_BANDS as u32).to_le_bytes());
        assert_eq!(&bytes[20..24], &(ROPE_DIM as u32).to_le_bytes());
        // Total length: header + each region's exact byte count (NO positions).
        let expect_len = 24 + len * (NOPE_DIM + NOPE_BANDS * 4 + ROPE_DIM * 2 + ih * 4);
        assert_eq!(bytes.len(), expect_len);

        let back = CorpusSnapshot::decode(&bytes).expect("decode");
        assert_eq!(back, snap);

        // Foreign magic / geometry mismatch → None.
        assert!(CorpusSnapshot::decode(b"XXXX....................").is_none());
        let mut bad = bytes.clone();
        bad[12] ^= 0xff; // corrupt NOPE_DIM
        assert!(CorpusSnapshot::decode(&bad).is_none());
        // Truncated payload → None.
        assert!(CorpusSnapshot::decode(&bytes[..bytes.len() - 1]).is_none());

        // Empty corpus round-trips too.
        let empty = CorpusSnapshot {
            index_head_dim: ih,
            len: 0,
            nope_i8: Vec::new(),
            nope_scale: Vec::new(),
            rope_bf: Vec::new(),
            keys: Vec::new(),
        };
        assert_eq!(CorpusSnapshot::decode(&empty.encode()).unwrap(), empty);
        Ok(())
    }

    /// Gate (Artifact C): a live gallery snapshotted and restored is
    /// byte-identical on the native durable set — the two-region cache, keys,
    /// AND the repacked sign index — independent of positions, and the injected
    /// positions are RECONSTRUCTED at a fresh base (`base + i·ratio`), NOT taken
    /// from the snapshot (which stores none). Straddles a growth (4 → 16).
    #[test]
    #[ignore]
    fn gallery_snapshot_restore_round_trip() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let ratio = 4u32; // CSA
        let mut g = FloatGallery::new(&dev, HD, IH, 4)?;
        let mut s = 77u64;
        // Live append uses the true group-start positions i·ratio.
        let mut appended = 0u32;
        for n in [3usize, 9] {
            let attn = rows(n, HD, &mut s);
            let keys = rows(n, IH, &mut s);
            let pos: Vec<u32> = (0..n as u32).map(|i| (appended + i) * ratio).collect();
            appended += n as u32;
            g.append_batch(
                &Tensor::from_vec(attn, (n, HD), &dev)?,
                &Tensor::from_vec(keys, (n, IH), &dev)?,
                &pos,
            )?;
        }
        let snap = g.snapshot()?;
        assert_eq!(snap.len, g.len());

        // Byte-exact codec on real data, then restore from the decoded blob at a
        // NEW absolute base (the resume frame) — positions are reconstructed.
        let blob = snap.encode();
        let snap2 = CorpusSnapshot::decode(&blob).expect("decode");
        assert_eq!(snap2, snap);
        let new_base = 10_000u32;
        let recon_pos: Vec<u32> = (0..snap2.len as u32)
            .map(|i| new_base + i * ratio)
            .collect();
        let r = FloatGallery::from_snapshot(&dev, &snap2, &recon_pos)?;
        assert_eq!(r.len(), g.len());

        // Durable buffers must match bit-for-bit (independent of positions).
        let cmp_u8 = |a: &Tensor, b: &Tensor, label: &str| -> Result<()> {
            let av = a.flatten_all()?.to_vec1::<u8>()?;
            let bv = b.flatten_all()?.to_vec1::<u8>()?;
            assert_eq!(av, bv, "{label}");
            Ok(())
        };
        let live = g.len();
        cmp_u8(
            &g.nope_i8.narrow(0, 0, live)?.to_device(&Device::Cpu)?,
            &r.nope_i8.narrow(0, 0, live)?.to_device(&Device::Cpu)?,
            "nope_i8",
        )?;
        let f_ns_g = g
            .nope_scale
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let f_ns_r = r
            .nope_scale
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (&a, &b)) in f_ns_g.iter().zip(&f_ns_r).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "nope_scale[{i}]");
        }
        let rope_g = g
            .rope_bf
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        let rope_r = r
            .rope_bf
            .narrow(0, 0, live)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        for (i, (&a, &b)) in rope_g.iter().zip(&rope_r).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "rope_bf[{i}]");
        }
        let keys_g = g.scoring_keys()?.flatten_all()?.to_vec1::<f32>()?;
        let keys_r = r.scoring_keys()?.flatten_all()?.to_vec1::<f32>()?;
        for (i, (&a, &b)) in keys_g.iter().zip(&keys_r).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "keys[{i}]");
        }
        // Positions are the reconstructed ones (shifted to the new base), NOT
        // the originals.
        assert_eq!(
            r.positions()?.to_vec1::<u32>()?,
            recon_pos,
            "reconstructed pos"
        );
        assert_eq!(
            g.packed_signs()?.flatten_all()?.to_vec1::<u32>()?,
            r.packed_signs()?.flatten_all()?.to_vec1::<u32>()?,
            "signs (repacked from keys)"
        );
        Ok(())
    }

    /// The device sign-pack matches host bit packing exactly.
    #[test]
    #[ignore]
    fn sign_pack_matches_host() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 11u64;
        let n = 37usize;
        let vals = rows(n, IH, &mut s);
        let t = Tensor::from_vec(vals.clone(), (n, IH), &dev)?;
        let packed = sign_pack(&t)?.to_vec2::<u32>()?;
        for (r, row) in packed.iter().enumerate() {
            for (w, &word) in row.iter().enumerate() {
                let mut expect = 0u32;
                for b in 0..32 {
                    let d = w * 32 + b;
                    if d < IH && vals[r * IH + d] >= 0.0 {
                        expect |= 1 << b;
                    }
                }
                assert_eq!(word, expect, "row {r} word {w}");
            }
        }
        Ok(())
    }

    /// Gate (c): with the shortlist covering everything, the two-stage
    /// selection IS the full Indexer top-k; with a modest shortlist over
    /// structured (query-correlated) data it still recovers it exactly.
    #[test]
    #[ignore]
    fn two_stage_equals_full_indexer() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 5u64;
        let g_total = 500usize;
        let n_heads = 4usize;
        let top_k = 8usize;

        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        // A trained Indexer space makes relevance ≈ direction and its heads
        // are largely redundant on relevant content: model the query as a
        // shared direction plus per-head jitter, and plant relevant entries
        // along that shared direction. (Planting along the mean of fully
        // independent heads caps sign agreement at ~2/3 — not what a learned
        // relevance space looks like.)
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals.clone(), (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        let mut keys = rows(g_total, IH, &mut s);
        for (j, &e) in [7usize, 42, 137, 260, 401, 444, 471, 490]
            .iter()
            .enumerate()
        {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * (2.0 + j as f32 * 0.1) + lcg(&mut s) * 0.1;
            }
        }
        let attn = rows(g_total, HD, &mut s);
        let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();
        gal.append_batch(
            &Tensor::from_vec(attn, (g_total, HD), &dev)?,
            &Tensor::from_vec(keys, (g_total, IH), &dev)?,
            &pos,
        )?;

        let full = gal.full_indexer_top_k(&q, &weights, top_k)?;

        // Diagnostic: planted-entry agreement counts vs the random field.
        {
            let q_signs = sign_pack(&q)?;
            let counts = bdp_recall(&q_signs, &gal.packed_signs()?, IH)?.to_vec1::<u32>()?;
            let planted = [7usize, 42, 137, 260, 401, 444, 471, 490];
            let pc: Vec<u32> = planted.iter().map(|&e| counts[e]).collect();
            let mut rest: Vec<u32> = counts
                .iter()
                .enumerate()
                .filter(|(i, _)| !planted.contains(i))
                .map(|(_, &c)| c)
                .collect();
            rest.sort_unstable_by(|a, b| b.cmp(a));
            eprintln!("[diag] planted counts = {pc:?}");
            eprintln!("[diag] top random counts = {:?}", &rest[..8]);
        }

        // M = everything → identical by construction.
        let (sel_t, k) = gal.two_stage_select(&q, &weights, g_total, top_k)?;
        assert_eq!(k, top_k);
        let sel = sel_t.to_vec1::<u32>()?;
        assert_eq!(sel, full, "M=len must equal full top-k (two-stage {sel:?})");

        // Modest shortlist on structured data → still exact.
        let (sel_t, k) = gal.two_stage_select(&q, &weights, 64, top_k)?;
        assert_eq!(k, top_k);
        assert_eq!(
            sel_t.to_vec1::<u32>()?,
            full,
            "M=64 on structured corpus must recover the full top-k"
        );
        Ok(())
    }

    /// `two_stage_select_causal(.., n_causal = K)` over an N-entry gallery must be
    /// bit-identical to `two_stage_select` over a gallery holding only the first
    /// K of the same entries — the causal-prefix bound the batched prefill relies
    /// on (each entry's sign-pack / key is independent of the ones after it).
    #[test]
    #[ignore]
    fn two_stage_select_causal_matches_prefix_gallery() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 17u64;
        let n_full = 300usize;
        let k_causal = 173usize; // arbitrary non-boundary prefix
        let n_heads = 4usize;
        let top_k = 8usize;

        // Shared-direction query with per-head jitter (as in two_stage_equals_full).
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals, (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        // Plant a few relevant entries INSIDE the causal prefix so the shortlist
        // is non-trivial and exact.
        let mut keys = rows(n_full, IH, &mut s);
        for (j, &e) in [3usize, 40, 88, 150, 170].iter().enumerate() {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * (2.0 + j as f32 * 0.1) + lcg(&mut s) * 0.1;
            }
        }
        let attn = rows(n_full, HD, &mut s);
        let pos: Vec<u32> = (0..n_full as u32).map(|i| i * 4).collect();

        let mut gal_full = FloatGallery::new(&dev, HD, IH, 8)?;
        gal_full.append_batch(
            &Tensor::from_vec(attn.clone(), (n_full, HD), &dev)?,
            &Tensor::from_vec(keys.clone(), (n_full, IH), &dev)?,
            &pos,
        )?;

        // Prefix gallery: only the first K entries (identical bytes).
        let mut gal_prefix = FloatGallery::new(&dev, HD, IH, 8)?;
        gal_prefix.append_batch(
            &Tensor::from_vec(attn[..k_causal * HD].to_vec(), (k_causal, HD), &dev)?,
            &Tensor::from_vec(keys[..k_causal * IH].to_vec(), (k_causal, IH), &dev)?,
            &pos[..k_causal],
        )?;

        for &m in &[64usize, 1024] {
            let (a_t, ak) = gal_full.two_stage_select_causal(&q, &weights, m, top_k, k_causal)?;
            let (b_t, bk) = gal_prefix.two_stage_select(&q, &weights, m, top_k)?;
            assert_eq!(ak, bk, "k mismatch (m={m})");
            assert_eq!(
                a_t.to_vec1::<u32>()?,
                b_t.to_vec1::<u32>()?,
                "causal(K) over N-gallery must equal full over K-gallery (m={m})"
            );
        }
        Ok(())
    }

    /// Whole-prompt [`FloatGallery::batched_causal_select`] must be bit-identical
    /// to the per-token [`FloatGallery::two_stage_select_causal`] loop it replaces
    /// in the in-regime case (widest window ≤ shortlist, so two-stage == full
    /// top-k), across per-token causal bounds that grow with position.
    #[test]
    #[ignore]
    fn batched_causal_select_matches_per_token() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 23u64;
        let n_corpus = 200usize; // ≤ 1024 → in-regime (exact) path
        let n_tok = 40usize;
        let n_heads = 4usize;
        let top_k = 8usize;

        // Corpus keys, a few planted along a shared direction so top-k is
        // non-trivial; per-token queries share that direction with jitter.
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let mut keys = rows(n_corpus, IH, &mut s);
        for (j, &e) in [3usize, 30, 77, 120, 160].iter().enumerate() {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * (2.0 + j as f32 * 0.1) + lcg(&mut s) * 0.1;
            }
        }
        let attn = rows(n_corpus, HD, &mut s);
        let pos: Vec<u32> = (0..n_corpus as u32).map(|i| i * 4).collect();
        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        gal.append_batch(
            &Tensor::from_vec(attn, (n_corpus, HD), &dev)?,
            &Tensor::from_vec(keys, (n_corpus, IH), &dev)?,
            &pos,
        )?;

        let mut qv = Vec::with_capacity(n_tok * n_heads * IH);
        for _ in 0..n_tok {
            for _ in 0..n_heads {
                for d in 0..IH {
                    qv.push(shared[d] + lcg(&mut s) * 0.5);
                }
            }
        }
        let q_idx = Tensor::from_vec(qv, (n_tok, n_heads, IH), &dev)?;
        let wv: Vec<f32> = (0..n_tok * n_heads).map(|_| 0.5 + lcg(&mut s)).collect();
        let weights = Tensor::from_vec(wv, (n_tok, n_heads), &dev)?;

        // Causal bound grows with position (groups completing), capped at n_corpus.
        let n_visible: Vec<usize> = (0..n_tok).map(|t| ((t + 1) * 5).min(n_corpus)).collect();

        let (comp_idx, comp_cnt, _n) =
            gal.batched_causal_select_device(&q_idx, &weights, &n_visible, top_k)?;
        let comp_idx = comp_idx.to_vec2::<u32>()?;
        let comp_cnt = comp_cnt.to_vec1::<u32>()?;

        for t in 0..n_tok {
            let qt = q_idx.narrow(0, t, 1)?.reshape((n_heads, IH))?;
            let wt = weights.narrow(0, t, 1)?.reshape(n_heads)?;
            // top_m = n_corpus ≥ n_visible[t] → the m≥n (full top-k) branch.
            let (gids_t, k) =
                gal.two_stage_select_causal(&qt, &wt, n_corpus, top_k, n_visible[t])?;
            let expect: Vec<u32> = if k == 0 {
                Vec::new()
            } else {
                gids_t.to_vec1::<u32>()?
            };
            // The device select returns absolute gids ascending, MAX-padded past
            // `comp_cnt`; the first `comp_cnt[t]` must equal the per-token oracle.
            let got: Vec<u32> = comp_idx[t][..comp_cnt[t] as usize].to_vec();
            assert_eq!(comp_cnt[t] as usize, expect.len(), "token {t} count");
            assert_eq!(got, expect, "token {t} (n_visible={})", n_visible[t]);
        }
        Ok(())
    }

    /// Recall sweep on the structured corpus: the sign top-M contains the
    /// float top-k with recall → 1 as M grows (printed for inspection).
    #[test]
    #[ignore]
    fn recall_sweep_synthetic() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 9u64;
        let g_total = 2000usize;
        let n_heads = 4usize;
        let top_k = 8usize;

        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        // Shared-direction query with per-head jitter (see
        // `two_stage_equals_full_indexer`) — relevant entries planted along
        // it with moderate noise, so the recall curve has a realistic shape.
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals.clone(), (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;
        let mut keys = rows(g_total, IH, &mut s);
        for e in (0..g_total).step_by(97) {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * 1.5 + lcg(&mut s) * 0.5;
            }
        }
        let attn = rows(g_total, HD, &mut s);
        let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();
        gal.append_batch(
            &Tensor::from_vec(attn, (g_total, HD), &dev)?,
            &Tensor::from_vec(keys, (g_total, IH), &dev)?,
            &pos,
        )?;

        let full = gal.full_indexer_top_k(&q, &weights, top_k)?;
        let q_signs = sign_pack(&q)?;
        let counts = bdp_recall(&q_signs, &gal.packed_signs()?, IH)?;
        let bins = n_heads * IH + 1;
        let mut last_recall = 0.0;
        for m in [16usize, 32, 64, 128, 256, 512, 1024, 2000] {
            let ids = topm_select(&counts, m, bins)?.to_vec1::<u32>()?;
            let short: std::collections::HashSet<u32> = ids.into_iter().collect();
            let hit = full.iter().filter(|g| short.contains(g)).count();
            last_recall = hit as f32 / full.len() as f32;
            eprintln!("[recall] M={m:5}: {hit}/{} = {last_recall:.3}", full.len());
        }
        assert!(
            (last_recall - 1.0).abs() < f32::EPSILON,
            "M=len must have recall 1.0"
        );
        Ok(())
    }

    /// §L footprint-flat (attended set, runtime): as the corpus grows from 1k
    /// to 200k entries, what each query ACTUALLY ATTENDS stays bounded — the
    /// two-stage selector rescores at most `top_m ≤ 1024` shortlist keys and
    /// returns exactly `top_k` entries, regardless of corpus size. That fixed
    /// attended set — not the growing corpus — is the O(1)-error budget. The
    /// resident BDP-scan bytes (`packed_signs`) are `sign_words·4 B` per entry
    /// (index_head_dim/32 words), a tiny constant-per-entry that stays the
    /// minority of the corpus (vs `head_dim` attended + `index_head_dim`
    /// scoring floats) — the part that must remain resident is bounded small,
    /// the rest (`attn`/`keys`) is spillable and gathered per query at the same
    /// fixed shortlist/top-k width at any depth.
    #[test]
    #[ignore]
    fn attended_set_bounded_as_corpus_grows() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 21u64;
        let n_heads = 4usize;
        let top_k = 8usize;
        let top_m = 512usize;

        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals, (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        let n_sign_words = sign_words(IH);
        for &g_total in &[1_000usize, 50_000, 200_000] {
            let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
            let keys = rows(g_total, IH, &mut s);
            let attn = rows(g_total, HD, &mut s);
            let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();
            gal.append_batch(
                &Tensor::from_vec(attn, (g_total, HD), &dev)?,
                &Tensor::from_vec(keys, (g_total, IH), &dev)?,
                &pos,
            )?;

            let (sel_t, k) = gal.two_stage_select(&q, &weights, top_m, top_k)?;
            // The ATTENDED set is exactly top_k — flat across a 200× corpus.
            assert_eq!(k, top_k, "selected count must stay top_k at N={g_total}");
            assert_eq!(
                sel_t.dim(0)?,
                top_k,
                "selection tensor width must stay top_k at N={g_total}"
            );

            // Resident BDP bytes grow linearly but at the tiny sign rate; the
            // spillable float pair dwarfs it — so the must-stay-resident share
            // is a bounded-small fraction that shrinks as head_dim dominates.
            let resident_sign_bytes = g_total * n_sign_words * 4;
            let spillable_float_bytes = g_total * (HD + IH) * 4;
            assert!(
                resident_sign_bytes * 20 < spillable_float_bytes,
                "resident sign bytes ({resident_sign_bytes}) must stay a small \
                 fraction of the spillable float pair ({spillable_float_bytes}) at N={g_total}"
            );
        }
        Ok(())
    }

    /// §L(b) resident-corpus spill: past `HOT_ENTRY_CAP` the float pair moves
    /// to CPU RAM (the sign/pos index stays on the GPU), and selection is still
    /// EXACT — the two-stage top-k over a spilled corpus reproduces the full
    /// Indexer oracle, proving the on-demand gather feeds the kernel the same
    /// entries it would have found fully-resident.
    #[test]
    #[ignore]
    fn spilled_corpus_selects_exactly() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut s = 33u64;
        let g_total = HOT_ENTRY_CAP + 4_000; // straddle the spill threshold
        let n_heads = 4usize;
        let top_k = 8usize;

        // Shared-direction query; plant relevant entries along it at known ids.
        let shared: Vec<f32> = (0..IH).map(|_| lcg(&mut s)).collect();
        let q_vals: Vec<f32> = (0..n_heads)
            .flat_map(|_| {
                shared
                    .iter()
                    .map(|&v| v + lcg(&mut s) * 0.3)
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = Tensor::from_vec(q_vals, (n_heads, IH), &dev)?;
        let weights = Tensor::from_vec(vec![1.0f32; n_heads], n_heads, &dev)?;

        // Plant exactly top_k STRONGLY-aligned entries (some before, some after
        // the spill boundary) so the top-k is unambiguous — dominating the
        // random field — and both the oracle and the two-stage path must return
        // precisely this set. Strong alignment also guarantees BDP recall.
        let planted = [
            11usize,
            900,
            4_242,
            7_000,
            HOT_ENTRY_CAP + 7,
            HOT_ENTRY_CAP + 500,
            HOT_ENTRY_CAP + 1_500,
            HOT_ENTRY_CAP + 3_900,
        ];
        assert_eq!(planted.len(), top_k);
        let mut keys = rows(g_total, IH, &mut s);
        for &e in &planted {
            for d in 0..IH {
                keys[e * IH + d] = shared[d] * 6.0 + lcg(&mut s) * 0.05;
            }
        }
        let attn = rows(g_total, HD, &mut s);
        let pos: Vec<u32> = (0..g_total as u32).map(|i| i * 4).collect();

        let mut gal = FloatGallery::new(&dev, HD, IH, 8)?;
        // Append in two batches straddling the threshold so the spill fires
        // mid-stream, exactly as a growing conversation would trip it.
        let half = g_total / 2;
        gal.append_batch(
            &Tensor::from_vec(attn[..half * HD].to_vec(), (half, HD), &dev)?,
            &Tensor::from_vec(keys[..half * IH].to_vec(), (half, IH), &dev)?,
            &pos[..half],
        )?;
        gal.append_batch(
            &Tensor::from_vec(attn[half * HD..].to_vec(), (g_total - half, HD), &dev)?,
            &Tensor::from_vec(keys[half * IH..].to_vec(), (g_total - half, IH), &dev)?,
            &pos[half..],
        )?;

        // The corpus spilled; the index stayed resident.
        assert!(
            gal.is_spilled(),
            "corpus past HOT_ENTRY_CAP must have spilled"
        );
        assert!(
            !gal.keys.device().is_cuda(),
            "keys must be on CPU when spilled"
        );
        assert!(
            !gal.nope_i8.device().is_cuda()
                && !gal.nope_scale.device().is_cuda()
                && !gal.rope_bf.device().is_cuda(),
            "the two-region cache must be on CPU when spilled (bounded hot VRAM)"
        );
        assert!(gal.signs.device().is_cuda(), "signs must stay GPU-resident");
        assert!(gal.pos.device().is_cuda(), "pos must stay GPU-resident");

        // Selection is EXACT over the spilled corpus: the dominant planted set
        // is what both the oracle and the two-stage path return.
        let mut expected: Vec<u32> = planted.iter().map(|&e| e as u32).collect();
        expected.sort_unstable();
        let full = gal.full_indexer_top_k(&q, &weights, top_k)?;
        assert_eq!(full, expected, "oracle must return the planted top-k");
        let (sel_t, k) = gal.two_stage_select(&q, &weights, 1024, top_k)?;
        assert_eq!(k, top_k);
        assert_eq!(
            sel_t.to_vec1::<u32>()?,
            expected,
            "spilled two-stage top-k must equal the planted set (== oracle)"
        );

        // gather_selected feeds the reference path a compacted GPU pair of the
        // dense f32, reconstructed from the (spilled) two-region cache.
        let (comp, comp_pos) = gal.gather_selected(&sel_t)?;
        assert_eq!(comp.dims(), &[top_k, HD]);
        assert_eq!(comp_pos.dims(), &[top_k]);
        assert!(comp.device().is_cuda() && comp_pos.device().is_cuda());

        // gather_corpus re-heats the bounded two-region selection from the
        // spilled RAM tier back onto the GPU (the hot decode/prefill path).
        let (ni8, nsc, rbf, cpos) = gal.gather_corpus(&sel_t)?;
        assert_eq!(ni8.dims(), &[top_k, NOPE_DIM]);
        assert_eq!(nsc.dims(), &[top_k, NOPE_BANDS]);
        assert_eq!(rbf.dims(), &[top_k, ROPE_DIM]);
        assert_eq!(cpos.dims(), &[top_k]);
        assert!(
            ni8.device().is_cuda()
                && nsc.device().is_cuda()
                && rbf.device().is_cuda()
                && cpos.device().is_cuda(),
            "gather_corpus must return GPU tensors even when spilled"
        );
        Ok(())
    }
}
