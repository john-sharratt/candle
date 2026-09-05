//! A sequence's QSA index reconstructed from per-turn pieces.
//!
//! The live [`IndexCache`](super::indexer::IndexCache) is one growing buffer,
//! which is right while a sequence decodes: rows are appended in order and the
//! scorer hands the whole live prefix to cuBLAS as one transposed operand.
//!
//! A resumed conversation has no such buffer. Its index arrives as the sealed
//! pieces of the turns the projection selected — separately allocated, in
//! whatever order those turns were sealed, and **ragged**: the last row of a
//! piece covers fewer than `ratio` tokens, because a turn boundary does not land
//! on a block boundary. `T mod ratio` is left over, and the K/V's own
//! partial-chunk padding does not help because the index counts real tokens
//! while the chunk pads dead ones.
//!
//! So the reconstruction is **logical**. The pages stay where the turn records
//! put them and a descriptor table names them; nothing is concatenated. That
//! matters because the alternative — materialising the window — copies every
//! key of every selected turn on the step that reconstructs, which at
//! conversational depth is the largest copy on that path and buys the scorer
//! nothing it cannot already read (hot-path invariant 2b).
//!
//! # What "ragged" costs, and where it is paid
//!
//! Exactly one place: the per-query candidate count. Rows remain ordered by
//! token position, so "wholly below this query" is still a **prefix** however
//! wide each row is — only the mapping from a position to that prefix changes,
//! from `(pos + 1) / ratio` to a walk over the pages' widths. That walk happens
//! here, on the host, where the page table already lives, and the kernel is
//! handed the resulting `cnt` and never sees a width at all.

use candle::{DType, Device, Result, Tensor};

use super::indexer::IndexCache;

/// One page of a reconstructed index — a single turn's sealed rows.
#[derive(Debug, Clone)]
pub struct IndexPage {
    /// `[rows, head_dim]` F32, the turn's prepared block keys.
    pub keys: Tensor,
    /// Absolute token position this page's first row begins at.
    pub first_pos: usize,
    /// Tokens the page's LAST row covers, in `1..=ratio`. Every earlier row
    /// covers `ratio`. This is the whole of the ragged case: a turn of `T`
    /// tokens seals `ceil(T / ratio)` rows whose last one is `T - (rows-1)·ratio`
    /// wide.
    pub last_cells: usize,
    /// [`Self::keys`] channel-blocked as `[head_dim/4, rows, 4]` — the layout
    /// the scorer reads, built on first use and kept.
    ///
    /// Two layouts because they answer to different consumers: `keys` is what a
    /// row *is* and what the record stores, and this is what makes a warp's key
    /// read 512 contiguous bytes instead of 32 scattered ones. Building it is
    /// one pass over the page, done once when the page is first scored rather
    /// than per score.
    blocked: std::sync::OnceLock<Tensor>,
}

impl IndexPage {
    /// A page over `keys` (`[rows, head_dim]`) whose last row covers
    /// `last_cells` tokens, beginning at absolute token `first_pos`.
    pub fn new(keys: Tensor, first_pos: usize, last_cells: usize) -> Self {
        Self {
            keys,
            first_pos,
            last_cells,
            blocked: std::sync::OnceLock::new(),
        }
    }

    /// The channel-blocked keys, built on first use. See [`Self::blocked`].
    pub fn blocked(&self) -> Result<&Tensor> {
        if let Some(t) = self.blocked.get() {
            return Ok(t);
        }
        let (rows, dim) = self.keys.dims2()?;
        if dim % 4 != 0 {
            candle::bail!(
                "index page: head_dim {dim} is not a multiple of four — the scorer's \
                 channel-blocked staging is built in `float4` groups"
            );
        }
        let t = self
            .keys
            .reshape((rows, dim / 4, 4))?
            .transpose(0, 1)?
            .contiguous()?;
        Ok(self.blocked.get_or_init(|| t))
    }

    pub fn rows(&self) -> Result<usize> {
        self.keys.dim(0)
    }

    /// Tokens this page covers: full rows at `ratio`, plus the short last one.
    pub fn tokens(&self, ratio: usize) -> Result<usize> {
        let r = self.rows()?;
        Ok(if r == 0 {
            0
        } else {
            (r - 1) * ratio + self.last_cells
        })
    }
}

/// A logical index window over ordered pages.
pub struct PagedIndex {
    pages: Vec<IndexPage>,
    /// Exclusive prefix sum of page row counts; `page_first[P]` is the total.
    page_first: Vec<u32>,
    /// Exclusive prefix sum of page token counts, so a position resolves to a
    /// candidate prefix without walking rows.
    page_tokens: Vec<usize>,
    ratio: usize,
    device: Device,
    /// Device copies of the descriptor table, built once per window.
    keys_tbl: Option<Tensor>,
    first_tbl: Option<Tensor>,
    /// Each page's keys in the layout the scorer reads: `[head_dim/4, rows, 4]`
    /// — channel-blocked, so a warp scoring `n` consecutive candidates reads
    /// `n × 16` contiguous bytes per step.
    ///
    /// **This is the difference between 4 memory transactions and 32.** The
    /// record's layout is `[rows, head_dim]`, which is what a row means and what
    /// [`IndexCache::from_rows`] consumes, and in it consecutive candidates'
    /// keys are `head_dim × 4` bytes apart — so the warp's `float4` load
    /// scatters across 32 cache lines and the L1 hit rate measured **2%**.
    /// Blocking the channel axis outward puts the candidates adjacent again.
    ///
    /// Built once per window, beside the descriptor table, and then read for
    /// every decode step until the projection changes. It is a copy of the
    /// selected turns' keys — 16 MiB at 128K depth, one pass — which is a
    /// different thing from concatenating them per score: the pages stay
    /// separately addressed and the kernel still resolves each candidate to its
    /// own page.
    keys_t: Vec<Tensor>,
}

impl PagedIndex {
    /// Build a window over `pages`, which must be in ascending token order.
    ///
    /// Refuses a gap or an overlap rather than scoring across one: the
    /// candidate prefix is derived from the running token total, so a page
    /// whose `first_pos` does not continue the previous one would silently
    /// shift every later row's visibility.
    pub fn new(pages: Vec<IndexPage>, ratio: usize, device: &Device) -> Result<Self> {
        if ratio == 0 {
            candle::bail!("paged index: ratio must be positive");
        }
        let mut page_first = Vec::with_capacity(pages.len() + 1);
        let mut page_tokens = Vec::with_capacity(pages.len() + 1);
        page_first.push(0u32);
        page_tokens.push(0usize);
        // A page with no rows covers no tokens, so dropping it is exactly
        // equivalent — and keeping it is not. `page_first` would then hold a
        // duplicate, which breaks the strict ascent the page search's loop
        // invariant depends on: a candidate could resolve to the empty page and
        // read from a buffer with no rows in it.
        let mut pages: Vec<IndexPage> = pages;
        let mut keep = Ok(());
        pages.retain(|p| match p.rows() {
            Ok(r) => r > 0,
            Err(e) => {
                if keep.is_ok() {
                    keep = Err(e);
                }
                true
            }
        });
        keep?;
        let mut rows_acc = 0usize;
        let mut tok_acc = 0usize;
        for (i, p) in pages.iter().enumerate() {
            let r = p.rows()?;
            if r > 0 && !(1..=ratio).contains(&p.last_cells) {
                candle::bail!(
                    "paged index: page {i} declares {} cells in its last row, outside 1..={ratio}",
                    p.last_cells
                );
            }
            if p.first_pos != tok_acc {
                candle::bail!(
                    "paged index: page {i} starts at token {} but the pages before it cover \
                     {tok_acc} — a gap or an overlap would shift every later row's visibility",
                    p.first_pos
                );
            }
            rows_acc += r;
            tok_acc += p.tokens(ratio)?;
            page_first.push(rows_acc as u32);
            page_tokens.push(tok_acc);
        }
        Ok(Self {
            pages,
            page_first,
            page_tokens,
            ratio,
            device: device.clone(),
            keys_tbl: None,
            first_tbl: None,
            keys_t: Vec::new(),
        })
    }

    /// A window holding one live cache's rows — the degenerate single-page case,
    /// which is what a sequence that has not been reconstructed looks like.
    pub fn from_live(cache: &IndexCache, ratio: usize, device: &Device) -> Result<Self> {
        let rows = cache.live_rows()?;
        let last_cells = if rows.dim(0)? == 0 { 1 } else { ratio };
        Self::new(vec![IndexPage::new(rows, 0, last_cells)], ratio, device)
    }

    pub fn total_rows(&self) -> usize {
        *self.page_first.last().unwrap_or(&0) as usize
    }

    pub fn total_tokens(&self) -> usize {
        *self.page_tokens.last().unwrap_or(&0)
    }

    pub fn pages(&self) -> &[IndexPage] {
        &self.pages
    }

    /// Candidate rows wholly at or below `pos` — the ragged replacement for
    /// `(pos + 1) / ratio`.
    ///
    /// A row is a candidate when its whole span sits at or before the query, so
    /// this counts full pages while their token total fits, then the whole rows
    /// inside the page the position lands in. Identical to the uniform formula
    /// when every page is a multiple of `ratio`.
    pub fn candidates_at(&self, pos: usize) -> usize {
        let limit = pos + 1;
        // The last page whose token span ends at or before `limit`.
        let mut p = match self.page_tokens.binary_search(&limit) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        if p >= self.pages.len() {
            p = self.pages.len();
            return self.page_first[p] as usize;
        }
        let base_rows = self.page_first[p] as usize;
        let inside = limit - self.page_tokens[p];
        // Whole rows of this page that fit. The page's last row is short, so it
        // only counts when the position reaches the page's full token span.
        let page_rows = self.pages[p].rows().unwrap_or(0);
        let full_rows = page_rows.saturating_sub(1);
        let whole = (inside / self.ratio).min(full_rows);
        let all = if inside >= self.pages[p].tokens(self.ratio).unwrap_or(0) {
            page_rows
        } else {
            whole
        };
        base_rows + all
    }

    /// Materialise the descriptor table and the scorer's channel-blocked key
    /// staging on the device. Idempotent.
    ///
    /// The staging is `keys.reshape(rows, D/4, 4).transpose(0, 1)` made
    /// contiguous — see [`Self::keys_t`] for why the scorer wants that layout
    /// and not the record's.
    pub fn build_tables(&mut self) -> Result<()> {
        if self.keys_tbl.is_some() {
            return Ok(());
        }
        self.keys_t = Vec::with_capacity(self.pages.len());
        for p in &self.pages {
            let (rows, dim) = p.keys.dims2()?;
            if dim % 4 != 0 {
                candle::bail!(
                    "paged index: head_dim {dim} is not a multiple of four — the scorer's \
                     channel-blocked staging is built in `float4` groups"
                );
            }
            self.keys_t.push(
                p.keys
                    .reshape((rows, dim / 4, 4))?
                    .transpose(0, 1)?
                    .contiguous()?,
            );
        }
        let mut ptrs: Vec<i64> = Vec::with_capacity(self.pages.len());
        for t in &self.keys_t {
            ptrs.push(super::indexer::tensor_ptr(t)? as i64);
        }
        if ptrs.is_empty() {
            ptrs.push(0);
        }
        self.keys_tbl = Some(Tensor::from_vec(
            ptrs,
            (self.pages.len().max(1),),
            &self.device,
        )?);
        self.first_tbl = Some(Tensor::from_vec(
            self.page_first.clone(),
            (self.page_first.len(),),
            &self.device,
        )?);
        Ok(())
    }

    /// Score `q` against this window, writing `[rows, total_rows()]` into `out`.
    ///
    /// `qpos` are the queries' absolute token positions; the per-row candidate
    /// prefix is derived from them here so the kernel stays width-agnostic.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn score_rows(
        &mut self,
        q: &Tensor,
        qpos: &[usize],
        n_heads: usize,
        head_dim: usize,
        out: &Tensor,
        out_stride: usize,
        row_base: usize,
    ) -> Result<Vec<u32>> {
        use candle_kernels::simple::qsa_score_paged::run_qsa_score_paged;

        let (t, qh, qd) = q.dims3()?;
        if t != qpos.len() {
            candle::bail!("paged score: {t} rows against {} positions", qpos.len());
        }
        if qh != n_heads || qd != head_dim {
            candle::bail!(
                "paged score: queries are [{t}, {qh}, {qd}] against [_, {n_heads}, {head_dim}]"
            );
        }
        let cand: Vec<u32> = qpos.iter().map(|&p| self.candidates_at(p) as u32).collect();
        if t == 0 || self.total_rows() == 0 {
            return Ok(cand);
        }
        self.build_tables()?;
        let q = q.reshape((t * n_heads, head_dim))?.contiguous()?;
        let cnt = Tensor::from_vec(cand.clone(), (t,), &self.device)?;

        let Device::Cuda(cuda) = &self.device else {
            candle::bail!("paged score: runs on CUDA");
        };
        let stream = cuda.cuda_stream();
        let raw = stream.cu_stream() as *mut std::ffi::c_void;
        candle::set_kernel_breadcrumb("run_qsa_score_paged", file!(), line!());
        unsafe {
            run_qsa_score_paged(
                super::indexer::tensor_ptr(&q)? as *const f32,
                super::indexer::i64_ptr(self.keys_tbl.as_ref().unwrap())? as *const u64,
                super::indexer::u32_ptr(self.first_tbl.as_ref().unwrap())? as *const u32,
                super::indexer::u32_ptr(&cnt)? as *const u32,
                super::indexer::tensor_ptr(out)? as *mut f32,
                t as i32,
                n_heads as i32,
                head_dim as i32,
                self.total_rows() as i32,
                self.pages.len() as i32,
                out_stride as i64,
                row_base as i64,
                raw,
            );
        }
        Ok(cand)
    }

    /// The CPU oracle: the same expression, evaluated eagerly.
    ///
    /// Deliberately the naive form — one dot product at a time, in row order —
    /// so it agrees with the kernel only if the kernel is right, rather than by
    /// sharing an implementation with it.
    pub fn score_reference(
        &self,
        q: &Tensor,
        qpos: &[usize],
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let t = qpos.len();
        let n = self.total_rows();
        let qv = q.flatten_all()?.to_vec1::<f32>()?;
        let mut keys: Vec<f32> = Vec::with_capacity(n * head_dim);
        for p in &self.pages {
            keys.extend(p.keys.flatten_all()?.to_vec1::<f32>()?);
        }
        let mut out = vec![-1e30f32; t * n];
        for (r, &pos) in qpos.iter().enumerate() {
            let valid = self.candidates_at(pos);
            for g in 0..valid.min(n) {
                let mut s = 0f32;
                for h in 0..n_heads {
                    let qb = (r * n_heads + h) * head_dim;
                    let mut d = 0f32;
                    for c in 0..head_dim {
                        d += qv[qb + c] * keys[g * head_dim + c];
                    }
                    s += d.max(0.0);
                }
                out[r * n + g] = s;
            }
        }
        Ok(out)
    }
}

/// One attention layer's index as a turn record carries it: the completed rows,
/// plus the tokens that had not yet completed a row when the seal ran.
///
/// **The open block is the whole reason this is not just an [`IndexPage`].** A
/// turn ends where its text ends, so `T mod ratio` tokens are carried in the
/// cache's open buffer rather than pooled into a row. Persisting only the
/// completed rows loses them, and the loss is not a rounding error: the live
/// cache's arithmetic is `n_blocks · ratio + n_open == T`, so a sequence
/// restored with `n_open = 0` stands at `n_blocks · ratio` while its K/V stands
/// at `T`. Every row appended afterwards is pooled over the wrong tokens, and
/// the scorer's `(pos + 1) / ratio` addresses the wrong block — for the rest of
/// the conversation, silently, because both sides remain internally consistent.
///
/// So the open rows travel too, and a resume rebuilds the cache exactly rather
/// than approximately.
#[derive(Debug, Clone)]
pub struct SealedIndex {
    /// The completed rows, as a scoring page.
    pub page: IndexPage,
    /// `[n_open, head_dim]` F32 — the raw, un-pooled rows of the trailing
    /// partial block. Zero rows when the turn happened to end on a boundary.
    pub open: Tensor,
}

impl SealedIndex {
    /// Tokens this record covers: the page's rows plus the open block.
    pub fn tokens(&self, ratio: usize) -> Result<usize> {
        Ok(self.page.tokens(ratio)? + self.open.dim(0)?)
    }
}

/// Wire version of the carried-state container. Version 2 carries each layer's
/// open block beside its completed rows.
const AUX_VERSION: u32 = 2;

/// Everything this architecture carries that is not a DeltaNet layer stack,
/// as one opaque blob for the turn record's auxiliary slot.
///
/// **A container, not a struct**, because the two things inside it answer to
/// different owners: the PLE window is the model's, and the index pages are one
/// per attention layer. Length-prefixing each section means a later carried
/// class can be appended without the persistence layer — which never parses
/// this — learning anything about it.
///
/// Sits beside the DeltaNet snapshot in the same record rather than in a stream
/// of its own: the two describe the same instant of the same sequence, and a
/// resume that installed one without the other would be precisely the
/// state-without-KV mismatch the whole path exists to remove.
pub fn encode_aux(ple: &[u8], layers: &[SealedIndex]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&AUX_VERSION.to_le_bytes());
    out.extend_from_slice(&(ple.len() as u32).to_le_bytes());
    out.extend_from_slice(ple);
    out.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for s in layers {
        let blob = encode_page(&s.page.keys, s.page.last_cells, &s.open)?;
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&blob);
    }
    Ok(out)
}

/// Read back what [`encode_aux`] wrote: the PLE bytes and one sealed index per
/// attention layer.
pub fn decode_aux(blob: &[u8], dev: &Device) -> Result<(Vec<u8>, Vec<SealedIndex>)> {
    let u32_at = |o: usize| -> Result<u32> {
        let b = blob
            .get(o..o + 4)
            .ok_or_else(|| candle::Error::Msg("aux blob: truncated".into()))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let version = u32_at(0)?;
    if version != AUX_VERSION {
        candle::bail!("aux blob: version {version} unknown (this build reads {AUX_VERSION})");
    }
    let ple_len = u32_at(4)? as usize;
    let ple_end = 8 + ple_len;
    let ple = blob
        .get(8..ple_end)
        .ok_or_else(|| candle::Error::Msg("aux blob: PLE section truncated".into()))?
        .to_vec();
    let n_pages = u32_at(ple_end)? as usize;
    let mut off = ple_end + 4;
    let mut pages = Vec::with_capacity(n_pages);
    for i in 0..n_pages {
        let len = u32_at(off)? as usize;
        off += 4;
        let section = blob
            .get(off..off + len)
            .ok_or_else(|| candle::Error::Msg(format!("aux blob: page {i} truncated")))?;
        pages.push(decode_page(section, dev)?);
        off += len;
    }
    Ok((ple, pages))
}

/// A layer's completed rows and its open block, as raw little-endian F32.
pub fn encode_page(keys: &Tensor, last_cells: usize, open: &Tensor) -> Result<Vec<u8>> {
    let (rows, dim) = keys.dims2()?;
    let (n_open, open_dim) = open.dims2()?;
    if n_open > 0 && open_dim != dim {
        candle::bail!("index page: open rows are [{n_open}, {open_dim}] against a [_, {dim}] page");
    }
    let vals = keys.flatten_all()?.to_vec1::<f32>()?;
    let open_vals = open.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(16 + (vals.len() + open_vals.len()) * 4);
    out.extend_from_slice(&(rows as u32).to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&(last_cells as u32).to_le_bytes());
    out.extend_from_slice(&(n_open as u32).to_le_bytes());
    for v in vals.iter().chain(open_vals.iter()) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// Read back what [`encode_page`] wrote.
pub fn decode_page(blob: &[u8], dev: &Device) -> Result<SealedIndex> {
    let u32_at = |o: usize| -> Result<u32> {
        let b = blob
            .get(o..o + 4)
            .ok_or_else(|| candle::Error::Msg("index page: blob truncated".into()))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let rows = u32_at(0)? as usize;
    let dim = u32_at(4)? as usize;
    let last_cells = u32_at(8)? as usize;
    let n_open = u32_at(12)? as usize;
    let n = rows * dim;
    let n_o = n_open * dim;
    let want = 16 + (n + n_o) * 4;
    if blob.len() < want {
        candle::bail!(
            "index page: blob is {} bytes but [{rows}, {dim}] with {n_open} open rows needs \
             {want}",
            blob.len()
        );
    }
    let mut vals = Vec::with_capacity(n);
    for i in 0..n {
        vals.push(f32::from_bits(u32_at(16 + i * 4)?));
    }
    let mut open_vals = Vec::with_capacity(n_o);
    for i in 0..n_o {
        open_vals.push(f32::from_bits(u32_at(16 + (n + i) * 4)?));
    }
    Ok(SealedIndex {
        page: IndexPage::new(
            Tensor::from_vec(vals, (rows, dim), dev)?.to_dtype(DType::F32)?,
            0,
            last_cells,
        ),
        open: Tensor::from_vec(open_vals, (n_open, dim), dev)?.to_dtype(DType::F32)?,
    })
}
