//! Builds synthetic window slots and corpus fixtures in the production byte
//! layout, with no substrate or model behind them.
//!
//! The rung-two harness needs slots whose bytes the kernel reads exactly as it
//! would read production slots. Each chunk's bands are encoded by the codec the
//! mirror also uses, so the expected values come from the stored bytes. The
//! builders place windows at chosen base positions and can mix formats per
//! band, which exercises the kernel's per-band dispatch. The CPU mirror of the
//! kernel's exponential, its RoPE trigonometry and its corpus quantisation live
//! here as well, so the oracle is written once.

use super::*;

pub struct SyntheticSlots {
    /// Flat u8 band arenas — per-(chunk, band) regions at `band_offsets`,
    /// each in that band's authored format (F8E4M3 rows or GGML blocks).
    pub bands: Tensor,
    /// `[n_chunks_total * KVHEAD_BYTES]` u8.
    pub kvheads: Tensor,
    /// `[n_chunks_total * 16]` u8 — TokenSlice array (all slots concatenated).
    pub slices: Tensor,
    /// `[num_slots * SLOT_HEADER_BYTES]` u8 — SlotHeader array.
    pub headers: Tensor,
    /// Per slot: (first chunk index, n_chunks, n_tokens).
    pub slot_meta: Vec<(usize, usize, usize)>,
    /// Per slot, per token: the exact stored values the kernel dequantizes
    /// (each band's storage round-trip applied) — the mirror's key source.
    pub stored: Vec<Vec<[f32; HEAD_DIM]>>,
}

#[cfg(feature = "cuda")]
impl SyntheticSlots {
    /// Build slots from per-slot pre-RoPE window latents (each `[n_tokens][512]`
    /// f32, values FP8-representable). Chunk `c` of a slot holds tokens
    /// `[32c, 32c+32)` at rope positions starting `32c`; the last chunk is the
    /// writer (its `len` excludes the incoming token).
    pub fn build(dev: &candle::Device, windows: &[Vec<[f32; HEAD_DIM]>]) -> Result<Self> {
        Self::build_based(dev, windows, &vec![0usize; windows.len()])
    }

    /// As [`Self::build`] but each slot's window is rope-based at `bases[slot]`
    /// — chunk `c` holds tokens `[32c, 32c+32)` at rope positions
    /// `bases[slot] + 32c`, so the writer (and thus the derived query position)
    /// sits at `bases[slot] + n_tokens`. Lets a benchmark place a 128-token
    /// window at a realistic depth (e.g. positions `[D-128, D)`), exercising
    /// the RoPE table's high-position range the way a deep context does.
    pub fn build_based(
        dev: &candle::Device,
        windows: &[Vec<[f32; HEAD_DIM]>],
        bases: &[usize],
    ) -> Result<Self> {
        let specs: Vec<Vec<[BandSpec; N_BANDS]>> = windows
            .iter()
            .map(|w| vec![[BandSpec::default(); N_BANDS]; w.len() / CHUNK + 1])
            .collect();
        Self::build_mixed(dev, windows, bases, &specs)
    }

    /// As [`Self::build_based`] but each `(chunk, band)` is authored in the
    /// given [`BandSpec`] format — FP8 rows or token-oriented GGML quant
    /// blocks with a non-unit outer scale — exercising the kernels' adaptive
    /// per-band window dispatch. The writer chunk (each slot's last) must be
    /// FP8: the kernels' fused/glue scatters write FP8 only.
    pub fn build_mixed(
        dev: &candle::Device,
        windows: &[Vec<[f32; HEAD_DIM]>],
        bases: &[usize],
        specs: &[Vec<[BandSpec; N_BANDS]>],
    ) -> Result<Self> {
        let ident = identity_pal_map();
        let maps: Vec<Vec<[u8; HEAD_DIM / 2]>> =
            specs.iter().map(|s| vec![ident; s.len()]).collect();
        Self::build_mapped(dev, windows, bases, specs, &maps)
    }

    /// As [`Self::build_mixed`] but each chunk also carries a pal_map: dims
    /// are routed to their assigned palette's arena in RANK order (the
    /// adaptive PalQuant layout), and the KvHead records carry the map. The
    /// writer chunk (each slot's last) must be identity-mapped FP8 — the
    /// scatters write the identity layout only.
    pub fn build_mapped(
        dev: &candle::Device,
        windows: &[Vec<[f32; HEAD_DIM]>],
        bases: &[usize],
        specs: &[Vec<[BandSpec; N_BANDS]>],
        maps: &[Vec<[u8; HEAD_DIM / 2]>],
    ) -> Result<Self> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        if bases.len() != windows.len()
            || specs.len() != windows.len()
            || maps.len() != windows.len()
        {
            candle::bail!(
                "build_mapped: {} bases / {} specs / {} maps for {} windows",
                bases.len(),
                specs.len(),
                maps.len(),
                windows.len()
            );
        }
        let ident = identity_pal_map();

        let cuda = match dev {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("SyntheticSlots requires a CUDA device"),
        };
        let stream = cuda.cuda_stream();

        let mut slot_meta = Vec::new();
        let mut chunk_base = 0usize;
        for (slot, w) in windows.iter().enumerate() {
            // The writer chunk must have room for the incoming token.
            let n_chunks = w.len() / CHUNK + 1;
            if specs[slot].len() != n_chunks {
                candle::bail!(
                    "build_mixed: slot {slot} has {} chunks but {} specs",
                    n_chunks,
                    specs[slot].len()
                );
            }
            if specs[slot][n_chunks - 1].iter().any(|s| s.fmt != 34) {
                candle::bail!("build_mapped: slot {slot} writer chunk must be FP8 (fmt 34)");
            }
            if maps[slot].len() != n_chunks {
                candle::bail!(
                    "build_mapped: slot {slot} has {n_chunks} chunks but {} maps",
                    maps[slot].len()
                );
            }
            if maps[slot][n_chunks - 1] != ident {
                candle::bail!("build_mapped: slot {slot} writer chunk must be identity-mapped");
            }
            slot_meta.push((chunk_base, n_chunks, w.len()));
            chunk_base += n_chunks;
        }
        let n_chunks_total = chunk_base;

        // Band arenas: per-(chunk, band) regions in the authored format, at
        // 16-byte-aligned offsets in one flat buffer. The same codec that
        // emits the bytes yields the stored (dequantized) values the mirror
        // keys against.
        let mut band_bytes: Vec<u8> = Vec::new();
        let mut band_offsets = vec![0usize; n_chunks_total * N_BANDS];
        let mut stored: Vec<Vec<[f32; HEAD_DIM]>> = Vec::with_capacity(windows.len());
        for (slot, w) in windows.iter().enumerate() {
            let (first_chunk, n_chunks, _) = slot_meta[slot];
            let mut slot_stored = vec![[0f32; HEAD_DIM]; w.len()];
            for c in 0..n_chunks {
                let t0 = c * CHUNK;
                let t1 = w.len().min(t0 + CHUNK);
                let toks: Vec<[f32; HEAD_DIM]> = w[t0..t1].to_vec();
                let (per_band, dec) = mapped_chunk_roundtrip(&specs[slot][c], &toks)?;
                for (band, bytes) in per_band.iter().enumerate() {
                    let off = band_bytes.len().next_multiple_of(16);
                    band_bytes.resize(off, 0);
                    band_bytes.extend_from_slice(bytes);
                    band_offsets[(first_chunk + c) * N_BANDS + band] = off;
                }
                for (t, dt) in dec.iter().enumerate() {
                    slot_stored[t0 + t] = *dt;
                }
            }
            stored.push(slot_stored);
        }
        let n_band_bytes = band_bytes.len().max(1);
        band_bytes.resize(n_band_bytes, 0);
        let bands = Tensor::from_vec(band_bytes, n_band_bytes, dev)?;
        let bands_addr = {
            let (storage, _) = bands.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<u8>()?;
                    let (p, _g) = slice.device_ptr(&stream);
                    p
                }
                _ => unreachable!(),
            }
        };

        // KvHead records: one per chunk (n_kv_head = 1). Identity band map,
        // per-band authored format tag + outer scale, v_* mirrors k_* (K≡V).
        let chunk_specs: Vec<[BandSpec; N_BANDS]> =
            specs.iter().flat_map(|s| s.iter().copied()).collect();
        let chunk_maps: Vec<[u8; HEAD_DIM / 2]> =
            maps.iter().flat_map(|m| m.iter().copied()).collect();
        let mut kvhead_bytes = vec![0u8; n_chunks_total * KVHEAD_BYTES];
        for chunk in 0..n_chunks_total {
            let rec = &mut kvhead_bytes[chunk * KVHEAD_BYTES..(chunk + 1) * KVHEAD_BYTES];
            // Single 4-bit pal_map over [0,HD/2) (K≡V — the old v_pal half is
            // absorbed into the one map).
            rec[..HEAD_DIM / 2].copy_from_slice(&chunk_maps[chunk]);
            for band in 0..N_BANDS {
                let spec = chunk_specs[chunk][band];
                let addr = bands_addr + band_offsets[chunk * N_BANDS + band] as u64;
                // Per-band block offsets, parameterized on N_BANDS to match the
                // <HD, NP> slot_types.cuh accessors: k_ptr @ HD/2, v_ptr @
                // HD/2+NB*8, k_fmt @ +NB*16, v_fmt @ +NB*17, k_scale @ +NB*18,
                // v_scale @ +NB*22.
                let kp = HEAD_DIM / 2 + band * 8;
                rec[kp..kp + 8].copy_from_slice(&addr.to_le_bytes());
                let vp = HEAD_DIM / 2 + N_BANDS * 8 + band * 8;
                rec[vp..vp + 8].copy_from_slice(&addr.to_le_bytes());
                rec[HEAD_DIM / 2 + N_BANDS * 16 + band] = spec.fmt;
                rec[HEAD_DIM / 2 + N_BANDS * 17 + band] = spec.fmt;
                let ks = HEAD_DIM / 2 + N_BANDS * 18 + band * 4;
                rec[ks..ks + 4].copy_from_slice(&spec.outer.to_le_bytes());
                let vs = HEAD_DIM / 2 + N_BANDS * 22 + band * 4;
                rec[vs..vs + 4].copy_from_slice(&spec.outer.to_le_bytes());
            }
        }
        let kvheads = Tensor::from_vec(kvhead_bytes, n_chunks_total * KVHEAD_BYTES, dev)?;
        let kvheads_addr = {
            let (storage, _) = kvheads.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<u8>()?;
                    let (p, _g) = slice.device_ptr(&stream);
                    p
                }
                _ => unreachable!(),
            }
        };

        // TokenSlice array: offset 0, len = tokens in chunk (writer excludes
        // the incoming token), rope = 32*chunk_in_slot.
        let mut slice_bytes = vec![0u8; n_chunks_total * 16];
        for (slot, w) in windows.iter().enumerate() {
            let (first_chunk, n_chunks, n_tokens) = slot_meta[slot];
            for c in 0..n_chunks {
                let g = first_chunk + c;
                let rec = &mut slice_bytes[g * 16..g * 16 + 16];
                let within = c * CHUNK; // token index within the slot (for len)
                let rope = bases[slot] + within; // absolute rope base of this chunk
                let len = n_tokens.saturating_sub(within).min(CHUNK);
                rec[0..2].copy_from_slice(&0u16.to_le_bytes()); // offset
                rec[2..4].copy_from_slice(&(len as u16).to_le_bytes());
                rec[4..8].copy_from_slice(&(rope as u32).to_le_bytes()); // rope
                let ka = kvheads_addr + (g * KVHEAD_BYTES) as u64;
                rec[8..16].copy_from_slice(&ka.to_le_bytes());
            }
            let _ = w;
        }
        let slices = Tensor::from_vec(slice_bytes, n_chunks_total * 16, dev)?;
        let slices_addr = {
            let (storage, _) = slices.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<u8>()?;
                    let (p, _g) = slice.device_ptr(&stream);
                    p
                }
                _ => unreachable!(),
            }
        };

        // SlotHeaders.
        let mut header_bytes = Vec::with_capacity(windows.len() * SLOT_HEADER_BYTES);
        for (slot, _) in windows.iter().enumerate() {
            let (first_chunk, n_chunks, _) = slot_meta[slot];
            SlotHeaderHost {
                n_slices: n_chunks as u32,
                write_slice: (n_chunks - 1) as u32, // writer = last
                slices_ptr: slices_addr + (first_chunk * 16) as u64,
                // Unused by the DeepSeek kernel, as is the rung: a latent
                // layer's frequencies are chosen by its kind, per launch.
                position_map_ptr: 0,
                rope_rung: 0,
            }
            .write(&mut header_bytes);
        }
        let n = header_bytes.len();
        let headers = Tensor::from_vec(header_bytes, n, dev)?;

        Ok(Self {
            bands,
            kvheads,
            slices,
            headers,
            slot_meta,
            stored,
        })
    }

    /// Device address of the `SlotHeader` array — the raw-pointer form
    /// [`paged_latent_decode_raw`] wants, so a benchmark can drive the
    /// non-committing decode path (no on-device write-len advance) and keep the
    /// synthetic arena static across a timed launch loop.
    pub fn header_device_ptr(&self) -> Result<u64> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let dev = match self.headers.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("SyntheticSlots headers must be CUDA-resident"),
        };
        let stream = dev.cuda_stream();
        let (storage, layout) = self.headers.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => {
                let slice = c.as_cuda_slice::<u8>()?;
                let (p, _guard) = slice.device_ptr(&stream);
                Ok(p + layout.start_offset() as u64)
            }
            _ => candle::bail!("expected CUDA storage for headers"),
        }
    }
}

// ─── CPU mirror oracle (a): the kernel's exact arithmetic on the host ────────

/// The kernel-side `fast_exp` e^x, mirrored **instruction-for-instruction**
/// from the PTX variant that actually runs on SM80+
/// (`exp_f32_high_softmax_ptx`, fast_exp.cuh — `FAST_EXP_USE_PTX=1`): its
/// FMA-tuned cubic coefficients differ from the C-source constants, its Horner
/// steps are explicit `fma.rn`, and `cvt.rmi` is floor. Every op here maps 1:1
/// to that asm, so the replica is bit-exact. (The archive also compiles with
/// `-fmad=false`, so *implicit* contraction never appears anywhere else in the
/// kernel — plain mul/add mirrors plain mul/add.)
#[allow(dead_code)] // CPU mirror-oracle for the decode-kernel tests
pub fn ds_exp_mirror(x: f32) -> f32 {
    let log2_e = f32::from_bits(0x3FB8_AA3B);
    let c3 = f32::from_bits(0x3D9D_9653);
    let c2 = f32::from_bits(0x3E69_1E05);
    let c1 = f32::from_bits(0x3F31_F70E);
    let xc = x.max(-88.0);
    let x2 = xc * log2_e;
    let xi = x2.floor(); // cvt.rmi
    let xf = x2 - xi;
    let mut poly = xf.mul_add(c3, c2); // fma.rn xf·C3 + C2
    poly = poly.mul_add(xf, c1); //       poly·xf + C1
    poly = poly.mul_add(xf, 1.0); //      poly·xf + 1
    let scale = f32::from_bits((((xi as i32) + 127) << 23) as u32);
    poly * scale
}

/// Per-band int8 quantization: returns (int8 values, band scales) exactly as
/// the kernel computes them (max-abs per 128-dim band, `mx / 127` — IEEE
/// division, matching the kernel's `__fdiv_rn(mx, 127.f)`; the attention
/// kernels run under `--use_fast_math`, so the division is written as the
/// explicit round-to-nearest intrinsic — zero→1, then the requant divides
/// by `s` via `__frcp_rn(s)` = `1.0/s`, round-nearest-even, clamp ±127).
#[allow(dead_code)] // reference two-region quantizer for the mirror tests
pub fn quant_bands(v: &[f32; HEAD_DIM]) -> ([i8; HEAD_DIM], [f32; N_BANDS]) {
    let mut q = [0i8; HEAD_DIM];
    let mut scales = [1.0f32; N_BANDS];
    for (band, scale) in scales.iter_mut().enumerate() {
        let lo = band * SUB_DIM;
        let src = &v[lo..lo + SUB_DIM];
        let mx = src.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let mut s = mx / 127.0;
        if s == 0.0 {
            s = 1.0;
        }
        *scale = s;
        let inv = 1.0 / s;
        for (d, &x) in src.iter().enumerate() {
            q[lo + d] = (x * inv).clamp(-127.0, 127.0).round_ties_even() as i32 as i8;
        }
    }
    (q, scales)
}

/// CPU mirror of `latent_build_corpus_cache_kernel` (the two-region position-free
/// cache, review items B+D): NOPE bands `[0,NOPE_DIM)` → int8 with a per-band
/// component-amax scale; ROPE bands `[NOPE_DIM,HEAD_DIM)` → BF16 pre-rotation.
/// Byte-exact against the kernel — `f32→bf16` is round-to-nearest-even on both
/// sides, and the nope int8 is the same `mx/127`, RNE, clamp path as
/// [`quant_bands`]'s nope arm.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // reference two-region cache builder for the mirror tests
pub fn build_corpus_cache(
    v: &[f32; HEAD_DIM],
) -> (
    [i8; NOPE_DIM],
    [f32; NOPE_DIM / SUB_DIM],
    [half::bf16; ROPE_DIM],
) {
    const NOPE_BANDS: usize = NOPE_DIM / SUB_DIM;
    let mut nope = [0i8; NOPE_DIM];
    let mut scale = [1.0f32; NOPE_BANDS];
    for (band, band_scale) in scale.iter_mut().enumerate() {
        let lo = band * SUB_DIM;
        let src = &v[lo..lo + SUB_DIM];
        let mx = src.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let mut s = mx / 127.0;
        if s == 0.0 {
            s = 1.0;
        }
        *band_scale = s;
        let inv = 1.0 / s;
        for (d, &x) in src.iter().enumerate() {
            nope[lo + d] = (x * inv).clamp(-127.0, 127.0).round_ties_even() as i32 as i8;
        }
    }
    let rope: [half::bf16; ROPE_DIM] =
        std::array::from_fn(|d| half::bf16::from_f32(v[NOPE_DIM + d]));
    (nope, scale, rope)
}

/// One key's contribution to the logit: Σ_band (int32 dot) · sQ · sK, summed
/// in ascending band order — the kernel's per-band float accumulation chain.
#[allow(dead_code)] // CPU logit mirror-oracle for the decode-kernel tests
pub fn mirror_logit(
    q_i8: &[i8; HEAD_DIM],
    q_s: &[f32; N_BANDS],
    k_i8: &[i8; HEAD_DIM],
    k_s: &[f32; N_BANDS],
) -> f32 {
    let mut total = 0.0f32;
    for band in 0..N_BANDS {
        let lo = band * SUB_DIM;
        let mut acc: i32 = 0;
        for d in lo..lo + SUB_DIM {
            acc += (q_i8[d] as i32) * (k_i8[d] as i32);
        }
        total += (acc as f32) * q_s[band] * k_s[band];
    }
    total
}
