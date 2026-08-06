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

/// Words per packed sign row.
fn sign_words(dim: usize) -> usize {
    dim.div_ceil(32)
}

/// Entry count past which the float corpus pair (`attn` + `keys`) spills from
/// the GPU (hot) to CPU RAM (warm). The `signs` + `pos` index stays GPU-
/// resident at any depth (the BDP scan reads all of it, and it is tiny —
/// `sign_words·4 + 4` bytes/entry). Below the threshold a gallery is fully
/// hot, so short conversations and the reference paths pay nothing; beyond it
/// the resident footprint is bounded to the index while the corpus grows in
/// RAM and the bounded shortlist/selection is gathered back per query. At
/// 8192 entries (`ratio 4` ⇒ ~32k tokens) the hot float pair is ≤16 MB/gallery.
const HOT_ENTRY_CAP: usize = 8192;

/// Compressed-corpus pair with a packed sign index. The `signs`/`pos` index is
/// always GPU-resident; the `attn`/`keys` float pair spills to CPU RAM past
/// [`HOT_ENTRY_CAP`] entries (`spilled`), keeping the resident footprint
/// bounded at unbounded depth (§L). Grows by doubling.
pub struct FloatGallery {
    attn: Tensor,  // [cap, head_dim] f32, pre-RoPE — GPU while hot, CPU when spilled
    keys: Tensor,  // [cap, index_head_dim] f32, pre-RoPE — GPU while hot, CPU when spilled
    signs: Tensor, // [cap, sign_words] u32 — always GPU
    pos: Tensor,   // [cap] u32 group-start positions — always GPU
    len: usize,
    cap: usize,
    head_dim: usize,
    index_head_dim: usize,
    device: Device,
    /// True once `attn`/`keys` have moved to CPU RAM (the warm tier).
    spilled: bool,
}

impl FloatGallery {
    pub fn new(
        device: &Device,
        head_dim: usize,
        index_head_dim: usize,
        initial_cap: usize,
    ) -> Result<Self> {
        let cap = initial_cap.max(1);
        Ok(Self {
            attn: Tensor::zeros((cap, head_dim), DType::F32, device)?,
            keys: Tensor::zeros((cap, index_head_dim), DType::F32, device)?,
            signs: Tensor::zeros((cap, sign_words(index_head_dim)), DType::U32, device)?,
            pos: Tensor::zeros(cap, DType::U32, device)?,
            len: 0,
            cap,
            head_dim,
            index_head_dim,
            device: device.clone(),
            spilled: false,
        })
    }

    /// Whether the float corpus pair currently lives in CPU RAM (warm tier).
    pub fn is_spilled(&self) -> bool {
        self.spilled
    }

    /// The device the `attn`/`keys` float pair currently lives on (CPU once
    /// spilled). `signs`/`pos` are always on [`Self::device`].
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

    /// The live attended-entry view `[len, head_dim]` — what the decode
    /// kernel's compressed pass gathers from (`comp_ptr`).
    pub fn attn_entries(&self) -> Result<Tensor> {
        self.attn.narrow(0, 0, self.len.max(1))
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
        self.attn = grow(&self.attn, self.head_dim, DType::F32, &fdev)?;
        self.keys = grow(&self.keys, self.index_head_dim, DType::F32, &fdev)?;
        self.signs = grow(
            &self.signs,
            sign_words(self.index_head_dim),
            DType::U32,
            &self.device,
        )?;
        self.pos = grow(&self.pos, 0, DType::U32, &self.device)?;
        self.cap = new_cap;
        Ok(())
    }

    /// Move the float corpus pair (`attn`/`keys`) from the GPU to CPU RAM once
    /// the entry count crosses [`HOT_ENTRY_CAP`]. One-way: the corpus only
    /// grows, and re-heating a whole spilled pair would defeat the bound —
    /// per-query gathers pull the bounded working set back instead.
    fn maybe_spill(&mut self, prospective_len: usize) -> Result<()> {
        if self.spilled || prospective_len <= HOT_ENTRY_CAP || !self.device.is_cuda() {
            return Ok(());
        }
        // Move the live prefix to CPU (the capacity tail is zeros — reallocate
        // at the current cap so the CPU buffers match the GPU layout).
        self.attn = self.attn.to_device(&Device::Cpu)?;
        self.keys = self.keys.to_device(&Device::Cpu)?;
        self.spilled = true;
        Ok(())
    }

    /// Append `n` completed groups: attended rows `[n, head_dim]` f32, scoring
    /// keys `[n, index_head_dim]` f32 (both pre-RoPE, device), and their
    /// group-start positions. Sign bits are packed on-device.
    #[cfg(feature = "cuda")]
    pub fn append_batch(
        &mut self,
        attn_rows: &Tensor,
        key_rows: &Tensor,
        positions: &[u32],
    ) -> Result<()> {
        let (n, hd) = attn_rows.dims2()?;
        if n == 0 {
            return Ok(());
        }
        let (kn, kd) = key_rows.dims2()?;
        if hd != self.head_dim || kd != self.index_head_dim || kn != n || positions.len() != n {
            candle::bail!(
                "append_batch shape mismatch: attn {:?}, keys {:?}, pos {}",
                attn_rows.dims(),
                key_rows.dims(),
                positions.len()
            );
        }
        // Sign-pack the new keys on the GPU (the incoming rows are GPU) BEFORE
        // any spill moves the float pair — the sign index stays GPU-resident.
        let new_signs = sign_pack(key_rows)?;
        let pos_t = Tensor::from_vec(positions.to_vec(), n, &self.device)?;

        self.grow_to(self.len + n)?;
        self.maybe_spill(self.len + n)?;
        // The float pair may now live on CPU; place the incoming rows on the
        // same tier before writing them in.
        let fdev = self.float_device();
        let attn_rows = attn_rows.to_device(&fdev)?;
        let key_rows = key_rows.to_device(&fdev)?;
        self.attn.slice_set(&attn_rows, 0, self.len)?;
        self.keys.slice_set(&key_rows, 0, self.len)?;
        self.pos.slice_set(&pos_t, 0, self.len)?;
        self.signs.slice_set(&new_signs, 0, self.len)?;
        self.len += n;
        Ok(())
    }

    /// Gather the `attn` rows and positions for `gids` (absolute entry ids,
    /// GPU u32) into a COMPACTED GPU pair — `(attn [k, head_dim], pos [k])` —
    /// regardless of which tier the corpus lives on. The kernel then walks the
    /// selection densely (`comp_idx = 0..k`). The only work that scales with
    /// depth is the BDP scan over the resident sign index; this gather touches
    /// exactly the `k` selected rows, so it is bounded at any corpus size.
    #[cfg(feature = "cuda")]
    pub fn gather_selected(&self, gids: &Tensor) -> Result<(Tensor, Tensor)> {
        let pos = self.positions()?.index_select(gids, 0)?; // GPU
        let attn = if self.spilled {
            // Float pair is on CPU: read the k indices back (bounded), gather
            // on CPU, upload the compacted rows.
            let gids_cpu = gids.to_device(&Device::Cpu)?;
            self.attn
                .narrow(0, 0, self.len.max(1))?
                .index_select(&gids_cpu, 0)?
                .to_device(&self.device)?
        } else {
            self.attn_entries()?.index_select(gids, 0)?
        };
        Ok((attn.contiguous()?, pos.contiguous()?))
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
        if self.len == 0 || top_k == 0 {
            return Ok((Tensor::zeros(1, DType::U32, &self.device)?, 0));
        }
        // The bitonic argsort caps at 1024 columns, which bounds the rescore
        // width — the recall shortlist itself is selected by the exact
        // histogram top-M (any corpus size).
        let m = top_m.clamp(1, 1024).min(self.len);

        // Stage 1 — recall: shortlist by packed-sign agreement.
        let shortlist: Tensor = if m >= self.len {
            Tensor::arange(0u32, self.len as u32, &self.device)?
        } else {
            let n_heads = q_idx.dim(0)?;
            let q_signs = sign_pack(q_idx)?;
            let counts = bdp_recall(&q_signs, &self.packed_signs()?, self.index_head_dim)?;
            topm_select(&counts, m, n_heads * self.index_head_dim + 1)?
        };

        // Stage 2 — precision: Indexer float score over the shortlist only.
        // The keys gather is tier-aware — from CPU RAM when spilled, in place
        // otherwise — and touches only the bounded shortlist.
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
                                                        // Ascending entry order (the kernel walks the selection in group
                                                        // order): entry ids are exact in f32 (< 2^24), so an f32 argsort
                                                        // yields the ascending permutation on-device.
        let asc = gids
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .arg_sort_last_dim(true)?
            .squeeze(0)?
            .contiguous()?;
        let gids_sorted = gids.index_select(&asc, 0)?;
        Ok((gids_sorted.contiguous()?, k))
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
            candle_kernels::simple::deepseek_bdp::run_deepseek_topm_select(
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
            candle_kernels::simple::deepseek_bdp::run_deepseek_sign_pack(
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
            candle_kernels::simple::deepseek_bdp::run_deepseek_bdp_recall(
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
            candle::bail!("deepseek_bdp_recall launch failed: cuda error {code}");
        }
    }
    Ok(counts)
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

    /// Gate (b): the corpus pair reads back bit-for-bit (float passthrough —
    /// the corpus's job is retrieval, not compression).
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
            for (i, (&a, &b)) in attn.iter().zip(&got_attn).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "attn[{i}]");
            }
            for (i, (&a, &b)) in keys.iter().zip(&got_keys).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "keys[{i}]");
            }
            let got_pos = g.positions()?.to_vec1::<u32>()?;
            assert_eq!(&got_pos[before..before + n], &pos[..]);
        }
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
            !gal.attn.device().is_cuda(),
            "attn must be on CPU when spilled"
        );
        assert!(
            !gal.keys.device().is_cuda(),
            "keys must be on CPU when spilled"
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

        // gather_selected feeds the kernel a compacted GPU pair of the right
        // shape, gathered from CPU RAM.
        let (comp, comp_pos) = gal.gather_selected(&sel_t)?;
        assert_eq!(comp.dims(), &[top_k, HD]);
        assert_eq!(comp_pos.dims(), &[top_k]);
        assert!(comp.device().is_cuda() && comp_pos.device().is_cuda());
        Ok(())
    }
}
