//! Holds the CPU reference for the latent attention and the quantisation error
//! budgets that justify the window's two-region format.
//!
//! The mirror replicates the kernel's tiling, quantisation, softmax, split
//! combine and de-rotation, so kernel output is compared against it at bit
//! level. A float sink-softmax reference gives the model-quality answer at int8
//! tolerance. The error-margin instrument shows that the rope tail must stay
//! BF16 while the non-rope dimensions tolerate FP8, and the position-free cache
//! test shows that rotating a stored entry at any position matches rotating the
//! original entry there.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    // Host mirrors of the latent kernels. Every one of these is a transcription:
    // the kernel's own loop bounds, its float literals to full width, and its
    // `fminf`/`fmaxf` NaN behaviour. Restyling them into iterator form or
    // trimming digits would break the line-by-line diff against the `.cu` that
    // makes a mismatch findable.
    #![allow(
        clippy::assign_op_pattern,
        clippy::excessive_precision,
        clippy::needless_range_loop,
        clippy::neg_cmp_op_on_partial_ord,
        clippy::type_complexity
    )]

    use super::*;

    use crate::models::rope_schedule::yarn_freqs;

    use candle::Device;

    use super::super::tests_geometry::tests::H;

    /// Calibration instrument (CPU, no GPU): computes the DeepSeek-V4 REFERENCE
    /// per-region quantization error margins — the rope tail (kept BF16) and the
    /// non-rope dims (FP8 E4M3, per-64 ue8m0 scale) — and the whole-band
    /// round-trip rel-L2 of every candidate format, documenting why the window's
    /// two-region format (FP8 nope ‖ BF16 rope) matches the reference: rope ≪
    /// nope margin, so the rope tail must stay BF16 while the nope dims tolerate
    /// FP8. Representative data = post-RMSNorm latent ≈ per-dim N(0,1),
    /// bf16-rounded (the write source is bf16). Prints a table; asserts only the
    /// two invariants (rope ≪ nope, Q8 ≈ rope margin, Q4 ≈ nope margin).
    #[test]
    fn latent_reference_error_margins() {
        const NT: usize = CHUNK; // 32 tokens (one chunk)
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        // Box–Muller standard normal.
        let mut gauss = || {
            let (u1, u2) = (u().max(1e-12), u());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        // True f32 band [NT][SUB_DIM], and its bf16-rounded source (what the
        // seal reads).
        let mut xf = vec![[0f32; SUB_DIM]; NT];
        let mut src = vec![[0f32; SUB_DIM]; NT];
        for t in 0..NT {
            for d in 0..SUB_DIM {
                let v = gauss() as f32;
                xf[t][d] = v;
                src[t][d] = half::bf16::from_f32(v).to_f32();
            }
        }
        let rel_l2 = |a: &[[f32; SUB_DIM]], b: &[[f32; SUB_DIM]]| -> f32 {
            let (mut num, mut den) = (0f64, 0f64);
            for t in 0..a.len() {
                for d in 0..SUB_DIM {
                    let e = (a[t][d] - b[t][d]) as f64;
                    num += e * e;
                    den += (b[t][d] as f64) * (b[t][d] as f64);
                }
            }
            (num / den).sqrt() as f32
        };

        // ── Reference ROPE margin: BF16 vs true f32 ──────────────────────────
        let eps_rope = rel_l2(&src, &xf);

        // ── Reference NOPE margin: FP8 E4M3, per-64 ue8m0 power-of-two scale,
        // vs true f32 (the reference's `act_quant(block=64, scale_fmt=ue8m0)`).
        // Reference FP8 block = 64 dims; a 32-dim band is smaller than the
        // reference block, so clamp the block to the band width (per-32 at
        // SUB_DIM=32 — same order of margin as per-64 for this Gaussian data).
        let blk_dim = SUB_DIM.min(64);
        let mut nope = vec![[0f32; SUB_DIM]; NT];
        for t in 0..NT {
            for blk in 0..(SUB_DIM / blk_dim) {
                let base = blk * blk_dim;
                let mut amax = 0f32;
                for d in base..base + blk_dim {
                    amax = amax.max(xf[t][d].abs());
                }
                // ue8m0: scale = 2^round(log2(amax/448)); guard amax=0.
                let scale = if amax > 0.0 {
                    (amax / 448.0).log2().round().exp2()
                } else {
                    1.0
                };
                for d in base..base + blk_dim {
                    nope[t][d] = e4m3_to_f32(f32_to_e4m3(xf[t][d] / scale)) * scale;
                }
            }
        }
        let eps_nope = rel_l2(&nope, &xf);

        // ── Per-format ladder rel-L2 (round-trip vs the bf16 source) ─────────
        // f16-scale formats self-scale (outer=1); INT8-scale/Q0 use 1/amax as a
        // representative searched outer.
        let band_amax = src
            .iter()
            .flat_map(|r| r.iter())
            .fold(0f32, |m, &v| m.max(v.abs()));
        let inv_amax = if band_amax > 0.0 {
            1.0 / band_amax
        } else {
            1.0
        };
        let fmts: &[(&str, u8, f32)] = &[
            ("Q8_0", 7, 1.0),
            ("Q8_KS", 10, 1.0),
            ("Q4_0", 15, 1.0),
            ("Q4_1", 16, 1.0),
            ("Q4_KS", 18, 1.0),
            ("Q3_0", 19, 1.0),
            ("Q3_1", 20, 1.0),
            ("Q2_S", 25, inv_amax),
            ("Q2_A", 26, inv_amax),
            ("Q1_S", 27, inv_amax),
            ("Q1_A", 29, inv_amax),
            ("Q0", 33, inv_amax),
        ];
        eprintln!("\n=== DeepSeek-V4 latent quant error margins (rel-L2) ===");
        eprintln!("REFERENCE  rope (BF16)          = {eps_rope:.5}");
        eprintln!("REFERENCE  nope (FP8 E4M3 /64)  = {eps_nope:.5}");
        eprintln!("--- ladder formats (round-trip vs bf16 source) ---");
        for &(name, tag, outer) in fmts {
            let spec = BandSpec { fmt: tag, outer };
            let (_, dec) = band_chunk_roundtrip(spec, &src).unwrap();
            eprintln!(
                "  {name:6} (tag {tag:2}) rel-L2 = {:.5}",
                rel_l2(&dec, &src)
            );
        }
        eprintln!("=======================================================\n");

        // Invariants the asymmetric ladder relies on.
        assert!(
            eps_rope < eps_nope * 0.25,
            "rope must be far tighter than nope"
        );
        assert!((0.001..0.01).contains(&eps_rope), "bf16 rope margin ~0.003");
        assert!((0.02..0.09).contains(&eps_nope), "e4m3 nope margin ~0.045");
    }

    pub(in super::super) struct Case {
        pub(in super::super) n_win: usize,
        pub(in super::super) window_size: usize,
        pub(in super::super) g_total: usize,
        pub(in super::super) ratio: usize,
        pub(in super::super) selected: Vec<usize>,
        pub(in super::super) num_splits: usize,
        pub(in super::super) zero_rope: bool,
    }

    pub(in super::super) struct Inputs {
        pub(in super::super) window: Vec<[f32; HEAD_DIM]>, // FP8-exact stored latents
        pub(in super::super) kv_new: [f32; HEAD_DIM],      // FP8-exact incoming latent
        pub(in super::super) q: Vec<[f32; HEAD_DIM]>,      // per head, bf16-exact
        pub(in super::super) comp: Vec<[f32; HEAD_DIM]>,   // bf16-exact compressed entries
        pub(in super::super) comp_pos: Vec<u32>,
        pub(in super::super) sinks: Vec<f32>,
        pub(in super::super) freqs: Vec<f32>,
    }

    /// Deterministic pseudo-random values: FP8-exact grid k*0.25, |k| ≤ 15.
    pub(in super::super) fn fp8_exact(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let k = ((*seed >> 33) % 31) as i64 - 15;
        k as f32 * 0.25
    }

    /// bf16-exact values (round-trip through half::bf16).
    pub(in super::super) fn bf16_exact(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (((*seed >> 33) % 2001) as f32 - 1000.0) / 500.0;
        half::bf16::from_f32(v).to_f32()
    }

    pub(in super::super) fn gen_inputs(case: &Case, seed: u64) -> Inputs {
        let mut s = seed;
        let window: Vec<[f32; HEAD_DIM]> = (0..case.n_win)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let kv_new: [f32; HEAD_DIM] = std::array::from_fn(|_| fp8_exact(&mut s));
        let q: Vec<[f32; HEAD_DIM]> = (0..H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let comp: Vec<[f32; HEAD_DIM]> = (0..case.g_total)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let comp_pos: Vec<u32> = (0..case.g_total).map(|g| (g * case.ratio) as u32).collect();
        let sinks: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs: Vec<f32> = if case.zero_rope {
            vec![0.0; ROPE_DIM / 2]
        } else {
            yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
                .into_iter()
                .map(|f| f as f32)
                .collect()
        };
        Inputs {
            window,
            kv_new,
            q,
            comp,
            comp_pos,
            sinks,
            freqs,
        }
    }

    /// The kernel's `rope_angle` (f64 reduction → quadrant + residual),
    /// replicated exactly — exact-rounded f64 ops are bit-identical across
    /// CPU and GPU.
    // The constants are the kernel's literals, character for character, so this
    // stays a line-by-line diff against the `.cu`. Two of them happen to be the
    // nearest f64 to TAU and 2/π (verified bit-identical to the `std` consts),
    // but 1/TAU on the line above has no `std` name — naming half of them would
    // cost the transcription and buy no precision.
    #[allow(clippy::approx_constant)]
    pub(in super::super) fn rope_angle(pos: u32, freq: f32) -> (f32, i32) {
        let mut a = pos as f64 * freq as f64;
        let t = (a * 0.159_154_943_091_895_35).floor();
        a -= t * 6.283_185_307_179_586;
        let q = (a * 0.636_619_772_367_581_4 + 0.5).floor();
        let r = (a - q * std::f64::consts::FRAC_PI_2) as f32;
        (r, (q as i32) & 3)
    }

    /// The kernel's `ds_sincos` polynomial, op-for-op (plain mul/add — the
    /// archive compiles `-fmad=false`).
    pub(in super::super) fn ds_sincos(r: f32, k: i32) -> (f32, f32) {
        let x2 = r * r;
        let mut sp_in = -1.951_529_589_1e-4_f32;
        sp_in = sp_in * x2 + 8.332_160_873_6e-3;
        sp_in = sp_in * x2 + -1.666_665_461_1e-1;
        let rt = r * x2;
        let sp = r + rt * sp_in;
        let mut cp_in = 2.443_315_711_809_948e-5_f32;
        cp_in = cp_in * x2 + -1.388_731_625_493_765e-3;
        cp_in = cp_in * x2 + 4.166_664_568_298_827e-2;
        let x4 = x2 * x2;
        let mut cp = 1.0 - 0.5 * x2;
        cp = cp + x4 * cp_in;
        match k {
            0 => (sp, cp),
            1 => (cp, -sp),
            2 => (-sp, -cp),
            _ => (-cp, sp),
        }
    }

    pub(in super::super) fn mirror_sincos(pos: u32, freq: f32) -> (f32, f32) {
        let (r, k) = rope_angle(pos, freq);
        ds_sincos(r, k)
    }

    /// The kernel's factored-table sin/cos, op-for-op: the hi/lo table entries
    /// are `mirror_sincos` of the split positions (bit-identical to the device
    /// builder's `rope_angle`/`ds_sincos` per the sincos probe), combined by
    /// the angle-addition identity in plain exact-rounded f32 — the kernel's
    /// `rope_lookup`.
    pub(in super::super) fn table_sincos(pos: u32, freq: f32) -> (f32, f32) {
        let hi = (pos >> ROPE_LO_BITS).min(ROPE_HI_DIM as u32 - 1) << ROPE_LO_BITS;
        let lo = pos & (ROPE_LO_DIM as u32 - 1);
        let (sh, ch) = mirror_sincos(hi, freq);
        let (sl, cl) = mirror_sincos(lo, freq);
        (sh * cl + ch * sl, ch * cl - sh * sl)
    }

    pub(in super::super) fn rope_vec(v: &mut [f32; HEAD_DIM], pos: u32, freqs: &[f32]) {
        for k in 0..ROPE_DIM / 2 {
            let (s, c) = table_sincos(pos, freqs[k]);
            let d = NOPE_DIM + 2 * k;
            let (x0, x1) = (v[d], v[d + 1]);
            v[d] = x0 * c - x1 * s;
            v[d + 1] = x0 * s + x1 * c;
        }
    }

    /// Reassembly round-trip: the POSITION-FREE two-region cache built ONCE
    /// (`build_corpus_cache`) dequantized and rotated at ANY position in the
    /// attention window reproduces roping the ORIGINAL entry at that position.
    /// The property the rope/nope split buys: the cache carries no baked position,
    /// so re-selecting an entry at a different window position needs no rebuild —
    /// the reader just rotates the stored BF16 rope bands at read time. (CPU-only.)
    #[test]
    fn corpus_cache_is_position_free() {
        // Standard RoPE frequencies + a random pre-RoPE latent.
        let freqs: Vec<f32> = (0..ROPE_DIM / 2)
            .map(|k| 1.0f32 / 10000f32.powf(2.0 * k as f32 / ROPE_DIM as f32))
            .collect();
        let mut s = 0x1234_5678u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let l0: [f32; HEAD_DIM] = std::array::from_fn(|_| next());

        // Build the position-free two-region cache ONCE; dequantize both regions.
        let (nope, scale, rope) = build_corpus_cache(&l0);
        let dequant: [f32; HEAD_DIM] = std::array::from_fn(|d| {
            if d < NOPE_DIM {
                nope[d] as f32 * scale[d / SUB_DIM]
            } else {
                rope[d - NOPE_DIM].to_f32()
            }
        });

        // Positions across the attention window (≤ 1M context cap).
        let positions = [0u32, 1, 31, 97, 4096, 65_536, 500_000, 1_000_000];
        for &pos in &positions {
            // Rope-at-load from the SAME stored bytes vs roping the original.
            let mut from_cache = dequant;
            rope_vec(&mut from_cache, pos, &freqs);
            let mut reference = l0;
            rope_vec(&mut reference, pos, &freqs);
            for d in 0..HEAD_DIM {
                // NOPE: int8 dequant error ≤ one band step. ROPE: bf16 round-trip
                // (~2⁻⁸ relative), preserved by the orthonormal rotation.
                let tol = if d < NOPE_DIM {
                    scale[d / SUB_DIM] + 1e-4
                } else {
                    reference[d].abs() * 0.02 + 2e-3
                };
                assert!(
                    (from_cache[d] - reference[d]).abs() <= tol,
                    "pos {pos} dim {d}: rope-at-load {} vs ref {} exceeds tol {tol}",
                    from_cache[d],
                    reference[d]
                );
            }
        }

        // NOPE bands are byte-identical to the generic per-band quant (never rope).
        let (ci_plain, _) = quant_bands(&l0);
        for d in 0..NOPE_DIM {
            assert_eq!(
                nope[d], ci_plain[d],
                "nope dim {d}: corpus quant diverges from plain per-band quant"
            );
        }
    }

    /// Stage 1 (B+D): the two-region cache builder is BYTE-EXACT against its CPU
    /// mirror — nope int8 + per-band amax scale, rope tail as BF16 pre-rotation.
    /// Proves the codec before decode/prefill are wired to it.
    #[test]
    #[ignore]
    fn corpus_cache_two_region_bytes() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        const NOPE_BANDS: usize = NOPE_DIM / SUB_DIM;
        let dev = Device::new_cuda(0)?;
        let cuda = match &dev {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        let stream = cuda.cuda_stream();

        // Random f32 entries (include a zero row → the scale zero→1 path).
        let g = 48usize;
        let mut s = 0x51ED_2701u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut entries: Vec<[f32; HEAD_DIM]> =
            (0..g).map(|_| std::array::from_fn(|_| next())).collect();
        entries[7] = [0.0; HEAD_DIM];
        let comp_flat: Vec<f32> = entries.iter().flat_map(|e| e.iter().copied()).collect();
        let comp = Tensor::from_vec(comp_flat, (g, HEAD_DIM), &dev)?;

        let nope_i8 = Tensor::zeros((g, NOPE_DIM), DType::U8, &dev)?;
        let nope_scale = Tensor::zeros((g, NOPE_BANDS), DType::F32, &dev)?;
        let rope_bf = Tensor::zeros((g, ROPE_DIM), DType::BF16, &dev)?;

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
        let comp_p = p!(&comp, f32);
        let ni8_p = p!(&nope_i8, u8);
        let nsc_p = p!(&nope_scale, f32);
        let rbf_p = p!(&rope_bf, half::bf16);
        unsafe {
            candle_kernels::paged_latent::run_latent_build_corpus_cache(
                comp_p as *const f32,
                ni8_p as *mut u8,
                nsc_p as *mut f32,
                rbf_p as *mut core::ffi::c_void,
                0,
                g as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            );
        }
        dev.synchronize()?;

        // int8 bytes come back as u8 (bit pattern); compare as i8.
        let got_nope = nope_i8.flatten_all()?.to_vec1::<u8>()?;
        let got_scale = nope_scale.flatten_all()?.to_vec1::<f32>()?;
        let got_rope = rope_bf.flatten_all()?.to_vec1::<half::bf16>()?;

        for (gi, e) in entries.iter().enumerate() {
            let (m_nope, m_scale, m_rope) = build_corpus_cache(e);
            for d in 0..NOPE_DIM {
                assert_eq!(
                    got_nope[gi * NOPE_DIM + d] as i8,
                    m_nope[d],
                    "entry {gi} nope[{d}]"
                );
            }
            for b in 0..NOPE_BANDS {
                assert_eq!(
                    got_scale[gi * NOPE_BANDS + b].to_bits(),
                    m_scale[b].to_bits(),
                    "entry {gi} scale[{b}]"
                );
            }
            for d in 0..ROPE_DIM {
                assert_eq!(
                    got_rope[gi * ROPE_DIM + d].to_bits(),
                    m_rope[d].to_bits(),
                    "entry {gi} rope[{d}]"
                );
            }
        }
        Ok(())
    }

    /// The full CPU mirror: replicates tiling, quantization, softmax phases,
    /// split ranges, combine + sink fold + de-rotation. Returns `[H][512]` f32
    /// (pre-BF16-store).
    pub(in super::super) fn mirror(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
    ) -> Vec<[f32; HEAD_DIM]> {
        mirror_fmt(case, inp, softmax_scale, None)
    }

    pub(in super::super) fn mirror_fmt(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
    ) -> Vec<[f32; HEAD_DIM]> {
        mirror_mapped(case, inp, softmax_scale, win_fmts)
    }

    /// Identity-layout mirror: each `(chunk, band)`'s stored values come from
    /// the same codec that authors the arena bytes (`band_chunk_roundtrip`),
    /// so the mirror keys off exactly what the kernel dequantizes.
    pub(in super::super) fn mirror_mapped(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
    ) -> Vec<[f32; HEAD_DIM]> {
        let n_win = case.n_win;
        let q_pos = n_win as u32;

        // Keys, in kernel order. Window token t lives at chunk t/32; the
        // writer's incoming token is the FP8 round-trip of kv_new.
        let n_chunks = n_win / CHUNK + 1;
        let mut chunk_lens: Vec<usize> = (0..n_chunks)
            .map(|c| n_win.saturating_sub(c * CHUNK).min(CHUNK))
            .collect();
        // slice_eff_len: writer (+1 for the scattered incoming token).
        *chunk_lens.last_mut().unwrap() += 1;

        // Stored window values: each (chunk, band)'s storage round-trip via
        // the SAME codec that authors the arena bytes (band_chunk_roundtrip),
        // so the mirror keys off exactly what the kernel dequantizes. The
        // scattered incoming token is always the writer chunk's FP8.
        let default_specs = vec![[BandSpec::default(); N_BANDS]; n_chunks];
        let specs = win_fmts.unwrap_or(&default_specs);
        let mut stored: Vec<[f32; HEAD_DIM]> = vec![[0.0; HEAD_DIM]; n_win];
        for c in 0..n_chunks {
            let t0 = c * CHUNK;
            let t1 = n_win.min(t0 + CHUNK);
            let toks: Vec<[f32; HEAD_DIM]> = inp.window[t0..t1].to_vec();
            let (_, dec) = mapped_chunk_roundtrip(&specs[c], &toks)
                .expect("mirror_mapped: codec rejected chunk spec");
            for (t, dt) in dec.iter().enumerate() {
                stored[t0 + t] = *dt;
            }
        }
        stored.push(std::array::from_fn(|d| {
            e4m3_to_f32(f32_to_e4m3(inp.kv_new[d]))
        }));

        // Tile list: per chunk, tiles of 8; then compressed tiles of 8.
        //
        // The kernel quantizes int8 from the roped f32 REGISTERS (full
        // precision), while the PV reads the bf16-STAGED latent — the mirror
        // must keep both, or zero-rope tests pass (fp8 ⊂ bf16, staging is
        // identity) while live-rope diverges.
        struct Key {
            pv: [f32; HEAD_DIM],  // post-RoPE, post-bf16-staging (PV read)
            k_i8: [i8; HEAD_DIM], // int8 from the roped f32 (QK read)
            k_s: [f32; N_BANDS],
            valid: bool,
        }
        let mut tiles: Vec<Vec<Key>> = Vec::new();
        for (c, &len) in chunk_lens.iter().enumerate() {
            let n_tiles = len.div_ceil(8);
            for t in 0..n_tiles {
                let mut keys = Vec::with_capacity(8);
                for w in 0..8 {
                    let within = t * 8 + w;
                    let mut key = Key {
                        pv: [0.0; HEAD_DIM],
                        k_i8: [0; HEAD_DIM],
                        k_s: [1.0; N_BANDS],
                        valid: false,
                    };
                    if within < len {
                        let pos = (c * CHUNK + within) as u32;
                        // Sliding-window + causal bound.
                        if pos <= q_pos && pos as i64 > q_pos as i64 - case.window_size as i64 {
                            let mut l = stored[c * CHUNK + within];
                            rope_vec(&mut l, pos, &inp.freqs);
                            let (ki, ks) = quant_bands(&l); // from roped f32
                            key.k_i8 = ki;
                            key.k_s = ks;
                            key.pv = std::array::from_fn(|d| half::bf16::from_f32(l[d]).to_f32());
                            key.valid = true;
                        }
                    }
                    keys.push(key);
                }
                tiles.push(keys);
            }
        }
        let n_comp_tiles = case.selected.len().div_ceil(8);
        for t in 0..n_comp_tiles {
            let mut keys = Vec::with_capacity(8);
            for w in 0..8 {
                let e = t * 8 + w;
                let mut key = Key {
                    pv: [0.0; HEAD_DIM],
                    k_i8: [0; HEAD_DIM],
                    k_s: [1.0; N_BANDS],
                    valid: false,
                };
                if e < case.selected.len() {
                    let gid = case.selected[e];
                    let pos = inp.comp_pos[gid];
                    // Causal guard: a compressed entry in the query's future is
                    // dropped (mirrors the kernel's `comp_pos <= q_pos` gate).
                    if pos <= q_pos {
                        // The kernel reads the persistent POSITION-FREE int8
                        // corpus cache (CorpusCache): nope bands raw, rope bands
                        // pre-rotation BF16 — `build_corpus_cache` IS the two-region
                        // cache builder's exact math. Dequantize both regions, then
                        // rotate the rope bands at this entry's assembled position
                        // (rope-at-load), then re-quantize for the int8 QK operand
                        // (`quant_bands`).
                        let mut l = inp.comp[gid];
                        let (nope, scale, rope) = build_corpus_cache(&l);
                        for d in 0..NOPE_DIM {
                            l[d] = nope[d] as f32 * scale[d / SUB_DIM];
                        }
                        for d in 0..ROPE_DIM {
                            l[NOPE_DIM + d] = rope[d].to_f32();
                        }
                        rope_vec(&mut l, pos, &inp.freqs);
                        let (ki, ks) = quant_bands(&l);
                        key.k_i8 = ki;
                        key.k_s = ks;
                        key.pv = std::array::from_fn(|d| half::bf16::from_f32(l[d]).to_f32());
                        key.valid = true;
                    }
                }
                keys.push(key);
            }
            tiles.push(keys);
        }

        // Q: rope at q_pos, band quant.
        let q_prep: Vec<([i8; HEAD_DIM], [f32; N_BANDS])> = inp
            .q
            .iter()
            .map(|qh| {
                let mut v = *qh;
                rope_vec(&mut v, q_pos, &inp.freqs);
                quant_bands(&v)
            })
            .collect();

        // Per-split flash accumulation over the split's tile range.
        let n_tiles = tiles.len();
        let num_splits = case.num_splits;
        let tiles_per_split = n_tiles.div_ceil(num_splits);
        let mut out = vec![[0.0f32; HEAD_DIM]; H];

        for (h, out_h) in out.iter_mut().enumerate() {
            let mut partials: Vec<(f32, f32, [f32; HEAD_DIM])> = Vec::new();
            for split in 0..num_splits {
                let lo = (split * tiles_per_split).min(n_tiles);
                let hi = (lo + tiles_per_split).min(n_tiles);
                let mut m = -1e38f32;
                let mut l = 0.0f32;
                let mut acc = [0.0f32; HEAD_DIM];
                for tile in &tiles[lo..hi] {
                    // Key quant + logits.
                    let mut sc = [0.0f32; 8];
                    let mut tile_max = -1e38f32;
                    for (t, key) in tile.iter().enumerate() {
                        let s = if key.valid {
                            mirror_logit(&q_prep[h].0, &q_prep[h].1, &key.k_i8, &key.k_s)
                                * softmax_scale
                        } else {
                            -1e38
                        };
                        sc[t] = s;
                        tile_max = tile_max.max(s);
                    }
                    let new_m = m.max(tile_max);
                    let alpha = ds_exp_mirror(m - new_m);
                    l *= alpha;
                    for a in acc.iter_mut() {
                        *a *= alpha;
                    }
                    for (t, key) in tile.iter().enumerate() {
                        let beta = if sc[t] > -1e37 {
                            ds_exp_mirror(sc[t] - new_m)
                        } else {
                            0.0
                        };
                        l += beta;
                        for d in 0..HEAD_DIM {
                            acc[d] = beta.mul_add(key.pv[d], acc[d]);
                        }
                    }
                    m = new_m;
                }
                partials.push((m, l, acc));
            }

            // Combine + sink fold + de-rotation (per dim).
            let mut gm = -1e38f32;
            for &(m, _, _) in &partials {
                gm = gm.max(m);
            }
            let sink = inp.sinks[h];
            let m_fin = gm.max(sink);
            let mut val = [0.0f32; HEAD_DIM];
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                let mut ll = 0.0f32;
                for &(m, l, ref a) in &partials {
                    if !(m > -1e37) {
                        continue;
                    }
                    let w = ds_exp_mirror(m - m_fin);
                    acc = a[d].mul_add(w, acc);
                    ll = l.mul_add(w, ll);
                }
                ll += ds_exp_mirror(sink - m_fin);
                val[d] = acc / ll.max(1e-10);
            }
            // Inverse rotation at q_pos.
            for k in 0..ROPE_DIM / 2 {
                let (s, c) = table_sincos(q_pos, inp.freqs[k]);
                let d = NOPE_DIM + 2 * k;
                let (x0, x1) = (val[d], val[d + 1]);
                val[d] = x0 * c + x1 * s;
                val[d + 1] = x1 * c - x0 * s;
            }
            *out_h = val;
        }
        out
    }

    /// Oracle (b): plain float sink-softmax attention over the same inputs.
    pub(in super::super) fn float_reference(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
    ) -> Vec<[f32; HEAD_DIM]> {
        let q_pos = case.n_win as u32;
        let mut keys: Vec<[f32; HEAD_DIM]> = Vec::new();
        for (t, w) in inp
            .window
            .iter()
            .chain(std::iter::once(&inp.kv_new))
            .enumerate()
        {
            let pos = t as u32;
            if pos <= q_pos && pos as i64 > q_pos as i64 - case.window_size as i64 {
                let mut l: [f32; HEAD_DIM] =
                    std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(w[d])));
                rope_vec(&mut l, pos, &inp.freqs);
                keys.push(l);
            }
        }
        for &gid in &case.selected {
            let pos = inp.comp_pos[gid];
            // Causal guard: a compressed entry in the query's future is dropped
            // (mirrors the kernel's `comp_pos <= q_pos` gate).
            if pos <= q_pos {
                let mut l = inp.comp[gid];
                rope_vec(&mut l, pos, &inp.freqs);
                keys.push(l);
            }
        }

        let mut out = vec![[0.0f32; HEAD_DIM]; H];
        for h in 0..H {
            let mut qv = inp.q[h];
            rope_vec(&mut qv, q_pos, &inp.freqs);
            let mut logits: Vec<f32> = keys
                .iter()
                .map(|k| {
                    let mut acc = 0.0f64;
                    for d in 0..HEAD_DIM {
                        acc += qv[d] as f64 * k[d] as f64;
                    }
                    (acc as f32) * softmax_scale
                })
                .collect();
            logits.push(inp.sinks[h]); // sink column
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f64> = logits.iter().map(|&v| ((v - m) as f64).exp()).collect();
            let z: f64 = exps.iter().sum();
            let mut val = [0.0f32; HEAD_DIM];
            for (k, key) in keys.iter().enumerate() {
                let p = (exps[k] / z) as f32;
                for d in 0..HEAD_DIM {
                    val[d] += p * key[d];
                }
            }
            for k in 0..ROPE_DIM / 2 {
                let (s, c) = table_sincos(q_pos, inp.freqs[k]);
                let d = NOPE_DIM + 2 * k;
                let (x0, x1) = (val[d], val[d + 1]);
                val[d] = x0 * c + x1 * s;
                val[d + 1] = x1 * c - x0 * s;
            }
            out[h] = val;
        }
        out
    }

    pub(in super::super) fn run_kernel(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
    ) -> Result<Vec<f32>> {
        run_kernel_fmt(case, inp, softmax_scale, None)
    }

    pub(in super::super) fn run_kernel_fmt(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
    ) -> Result<Vec<f32>> {
        run_kernel_mapped(case, inp, softmax_scale, win_fmts, None)
    }

    pub(in super::super) fn run_kernel_mapped(
        case: &Case,
        inp: &Inputs,
        softmax_scale: f32,
        win_fmts: Option<&[[BandSpec; N_BANDS]]>,
        win_maps: Option<&[[u8; HEAD_DIM / 2]]>,
    ) -> Result<Vec<f32>> {
        let dev = Device::new_cuda(0)?;
        let n_chunks = case.n_win / CHUNK + 1;
        let default_specs = vec![[BandSpec::default(); N_BANDS]; n_chunks];
        let specs = win_fmts.unwrap_or(&default_specs).to_vec();
        let ident = identity_pal_map();
        let default_maps = vec![ident; n_chunks];
        let maps = win_maps.unwrap_or(&default_maps).to_vec();
        let slots = SyntheticSlots::build_mapped(
            &dev,
            std::slice::from_ref(&inp.window),
            &[0usize],
            std::slice::from_ref(&specs),
            std::slice::from_ref(&maps),
        )?;

        let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
        let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let kvf: Vec<f32> = inp.kv_new.to_vec();
        let kv_new = Tensor::from_vec(kvf, (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
        let comp = Tensor::from_vec(compf, (case.g_total.max(1), HEAD_DIM), &dev)?;
        let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total.max(1), &dev)?;
        let max_sel = case.selected.len().max(1);
        let mut idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
        idx.resize(max_sel, u32::MAX);
        let comp_idx = Tensor::from_vec(idx, (1, max_sel), &dev)?;
        let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
        let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;

        let out = paged_latent_decode(
            &q,
            &slots.headers,
            &kv_new,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            case.window_size,
            case.num_splits,
            None,
        )?;
        // Kernel now emits F32; round to bf16 (reproducing the old in-kernel
        // `__float2bfloat16` store, RNE) so the mirror stays bit-exact — the
        // mirror side is `bf16_round_host`'d in `run_case`.
        out.to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    pub(in super::super) fn bf16_round_host(vals: &[f32]) -> Vec<f32> {
        vals.iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect()
    }

    /// Count cells where kernel and mirror differ by MORE than one bf16 code
    /// (with a `1e-4` absolute floor for near-zero cancellation cells, where
    /// bf16 codes are dense — a near-zero attention output accumulates several
    /// dense codes from ≤1-code per-key weight errors while its absolute error
    /// stays ~1e-4 · full-scale, i.e. negligible). Panics on any cell beyond
    /// that — a real bug moves cells by the value magnitude (~O(1)), far above
    /// the floor. Returns the off-by-one count for the ≥99.9% budget check.
    ///
    /// This is the tolerance the latent attention-OUTPUT mirror gates on: the
    /// kernel runs its int8 QK staging + softmax + dequant `/outer` under
    /// `--use_fast_math`, so a handful of cells land ≤1 bf16 code off the exact
    /// CPU mirror. The kernel is NOT worse than the mirror against the float
    /// reference (measured `kernel-vs-ref == mirror-vs-ref`); making the hot
    /// decode path IEEE-precise was measured ~8% slower and reverted, so an
    /// output-level ≤1-code tolerance is the right gate (these are attention
    /// OUTPUT tests, not codec raw-byte tests — those stay strict).
    pub(in super::super) fn check_within_one_code(kernel: &[f32], mirror: &[f32]) -> usize {
        let mut off = 0usize;
        for (i, (a, b)) in kernel.iter().zip(mirror).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            let (ca, cb) = (
                half::bf16::from_f32(*a).to_bits() as i32,
                half::bf16::from_f32(*b).to_bits() as i32,
            );
            assert!(
                (ca - cb).abs() <= 1 || (a - b).abs() <= 1e-4,
                "divergence beyond 1 bf16 code at {i} (head {} dim {}): {a} vs {b}",
                i / HEAD_DIM,
                i % HEAD_DIM
            );
            off += 1;
        }
        off
    }

    pub(in super::super) fn run_case(
        case: &Case,
        seed: u64,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(case, seed);
        let kernel = run_kernel(case, &inp, softmax_scale)?;
        let mirror_out: Vec<f32> = mirror(case, &inp, softmax_scale)
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        let reference: Vec<f32> = float_reference(case, &inp, softmax_scale)
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        Ok((kernel, bf16_round_host(&mirror_out), reference))
    }

    pub(in super::super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
}
