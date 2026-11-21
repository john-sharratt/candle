//! The compressed KV corpus each DeepSeek-V4-Flash sequence attends over, and how
//! its entries are selected.
//!
//! A `FloatGallery` holds one sequence's compressed entries: the attended latent in
//! a two-region int8 and bf16 cache, the Indexer's scoring keys, and packed sign
//! bits for recall. Selection runs in two stages. Sign-agreement recall shortlists
//! the whole corpus, then the learned Indexer score picks the top-k from that
//! shortlist, so the work per query stays bounded however deep the corpus grows.
//! Past `HOT_ENTRY_CAP` entries the latent and the keys move to host RAM and are
//! gathered back per query, while the sign index stays on the device. Append,
//! gather and selection each serve a whole wave in one launch.
//!
//! - `append`: appending a wave's completed groups to each gallery.
//! - `gather`: gathering each session's selected rows into a shared block.
//! - `recall`: top-M selection and sign-agreement recall.
//! - `select`: the two-stage selection and the Indexer score.
//! - `snapshot`: the durable form of a gallery.
//! - `tests_gather`, `tests_recall`, `tests_score_reduce`, `tests_selection`,
//!   `tests_snapshot`: tests.

use candle::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
use super::desc;
#[cfg(feature = "cuda")]
use super::scatter::{rows_scatter, RowRun};
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::Generation;

/// Words per packed sign row.
fn sign_words(dim: usize) -> usize {
    dim.div_ceil(32)
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

const NOPE_BANDS: usize = NOPE_DIM / SUB_DIM;

// 14

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
        let scope = desc::scope(&self.device)?;
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
        let scope = desc::scope(&self.device)?;
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
            // that is the pinned warm pool of docs/archived/kv_tier_migration.md: page-
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
            let staged =
                Tensor::from_vec(staged, n_bytes, &Device::Cpu)?.to_device(&self.device)?;

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
        // [s, n_corpus]
        let weighted = indexer_score_reduce(&scores, weights, Some(&vis))?;
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

mod append;
mod gather;
mod recall;
mod select;
mod snapshot;
#[cfg(test)]
mod tests_gather;
#[cfg(test)]
mod tests_recall;
#[cfg(test)]
mod tests_score_reduce;
#[cfg(test)]
mod tests_selection;
#[cfg(test)]
mod tests_snapshot;

pub use append::*;
pub use gather::*;
pub use recall::*;
pub use select::*;
pub use snapshot::*;
