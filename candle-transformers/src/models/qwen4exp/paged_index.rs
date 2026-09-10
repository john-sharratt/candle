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
#[cfg(feature = "cuda")]
use super::place::{PlacePage, Placement, PLACE_TILE_R};
#[cfg(feature = "cuda")]
use crate::models::qwen35::attention::RopeTables;

/// One page of a reconstructed index — a single turn's sealed rows.
///
/// **A page is position-free, and that is what makes it injectable.** Its rows
/// are the pooled, normed, roped block keys of one sealed unit; the rotation
/// they carry is the frame they were prepared in ([`Self::roped_base`]), not the
/// position they will be read at. A cache placing the page turns it through the
/// difference — see `IndexCache::push_page` and `models::qwen4exp::place`.
#[derive(Debug, Clone)]
pub struct IndexPage {
    /// `[rows, head_dim]` F32, the turn's prepared block keys.
    pub keys: Tensor,
    /// The absolute position [`Self::keys`] were **roped at**.
    ///
    /// Zero for a sealed page: the seal normalises it, so a record on disk holds
    /// no position at all and can be placed anywhere. Non-zero only while a page
    /// is still sitting in the cache that closed it, where it was roped at the
    /// place it already occupies and the placement rotation is the identity.
    ///
    /// This is the index's counterpart to a KV chunk's `rope_base`, and it is
    /// the whole of the difference between a page that can move and one that
    /// cannot.
    pub roped_base: usize,
    /// Tokens the page's LAST row covers, in `1..=ratio`. Every earlier row
    /// covers `ratio`. This is the whole of the ragged case: a turn of `T`
    /// tokens seals `ceil(T / ratio)` rows whose last one is `T - (rows-1)·ratio`
    /// wide.
    pub last_cells: usize,
}

impl IndexPage {
    /// A **position-free** page over `keys` (`[rows, head_dim]`) whose last row
    /// covers `last_cells` tokens — the sealed form, and what a record holds.
    pub fn new(keys: Tensor, last_cells: usize) -> Self {
        Self {
            keys,
            roped_base: 0,
            last_cells,
        }
    }

    /// A page whose rows were roped at `roped_base` rather than at zero — the
    /// live tail of a cache, lifted into a page where it already sits.
    pub fn at_frame(keys: Tensor, roped_base: usize, last_cells: usize) -> Self {
        Self {
            keys,
            roped_base,
            last_cells,
        }
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

/// A logical index window over placed pages — the reference harness the
/// `qsa_score_paged` kernel is gated against (`tests/qsa_paged_index_tests.rs`).
pub struct PagedIndex {
    pages: Vec<IndexPage>,
    /// Each page's absolute position, ascending and non-overlapping.
    bases: Vec<usize>,
    /// Exclusive prefix sum of page row counts; `page_first[P]` is the total.
    page_first: Vec<u32>,
    /// Each page's token width, resolved at construction where `ratio` is in
    /// hand.
    page_tokens: Vec<usize>,
    /// The position past the last page — where a live tail would open.
    end: usize,
    ratio: usize,
    device: Device,
    /// Device copies of the descriptor table, built once per window.
    keys_tbl: Option<Tensor>,
    first_tbl: Option<Tensor>,
    /// Every page rotated into the frame of the position it sits at and written
    /// in the layout the scorer reads: `[head_dim/4, rows, 4]` — channel-
    /// blocked, so a warp scoring `n` consecutive candidates reads `n × 16`
    /// contiguous bytes per step.
    ///
    /// **The blocking is the difference between 4 memory transactions and 32.**
    /// The record's layout is `[rows, head_dim]`, which is what a row means and
    /// what [`IndexCache::from_rows`] consumes, and in it consecutive
    /// candidates' keys are `head_dim × 4` bytes apart — so the warp's `float4`
    /// load scatters across 32 cache lines and the L1 hit rate measured **2%**.
    ///
    /// **The rotation is what makes the page placeable.** Its rows carry the
    /// frame they were roped in; this staging carries the frame they are read
    /// in. Both happen in the one pass the staging costs anyway.
    placed: Option<Placement>,
}

impl PagedIndex {
    /// Build a window over `(page, base)` pairs — each page and the absolute
    /// position it occupies.
    ///
    /// Bases must **ascend and not overlap**. A hole between two pages is legal
    /// and means exactly what it says: that span carries no index rows. It used
    /// to be refused, because the candidate prefix was derived from a running
    /// token total and a page that did not continue the previous one shifted
    /// every later row's visibility — positions are now read off the page, so
    /// there is nothing left to shift.
    pub fn new(placed: Vec<(IndexPage, usize)>, ratio: usize, device: &Device) -> Result<Self> {
        if ratio == 0 {
            candle::bail!("paged index: ratio must be positive");
        }
        // A page with no rows covers no tokens, so dropping it is exactly
        // equivalent — and keeping it is not. `page_first` would then hold a
        // duplicate, which breaks the strict ascent the page search's loop
        // invariant depends on: a candidate could resolve to the empty page and
        // read from a buffer with no rows in it.
        let mut keep = Ok(());
        let mut placed: Vec<(IndexPage, usize)> = placed;
        placed.retain(|(p, _)| match p.rows() {
            Ok(r) => r > 0,
            Err(e) => {
                if keep.is_ok() {
                    keep = Err(e);
                }
                true
            }
        });
        keep?;

        let mut pages = Vec::with_capacity(placed.len());
        let mut bases = Vec::with_capacity(placed.len());
        let mut page_first = Vec::with_capacity(placed.len() + 1);
        let mut page_tokens = Vec::with_capacity(placed.len());
        page_first.push(0u32);
        let mut rows_acc = 0usize;
        let mut end = 0usize;
        for (i, (p, base)) in placed.into_iter().enumerate() {
            let r = p.rows()?;
            if !(1..=ratio).contains(&p.last_cells) {
                candle::bail!(
                    "paged index: page {i} declares {} cells in its last row, outside 1..={ratio}",
                    p.last_cells
                );
            }
            if base < end {
                candle::bail!(
                    "paged index: page {i} opens at token {base} but the page before it runs to \
                     {end} — pages must ascend and may not overlap"
                );
            }
            let tokens = p.tokens(ratio)?;
            rows_acc += r;
            end = base + tokens;
            page_first.push(rows_acc as u32);
            page_tokens.push(tokens);
            bases.push(base);
            pages.push(p);
        }
        Ok(Self {
            pages,
            bases,
            page_first,
            page_tokens,
            end,
            ratio,
            device: device.clone(),
            keys_tbl: None,
            first_tbl: None,
            placed: None,
        })
    }

    /// A window holding one live cache's rows — the degenerate single-page case,
    /// which is what a sequence that has not been reconstructed looks like.
    pub fn from_live(cache: &IndexCache, ratio: usize, device: &Device) -> Result<Self> {
        let rows = cache.live_rows()?;
        let last_cells = if rows.dim(0)? == 0 { 1 } else { ratio };
        Self::new(vec![(IndexPage::new(rows, last_cells), 0)], ratio, device)
    }

    pub fn total_rows(&self) -> usize {
        *self.page_first.last().unwrap_or(&0) as usize
    }

    /// The position past the last page.
    pub fn total_tokens(&self) -> usize {
        self.end
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
        if limit > self.end {
            return self.total_rows();
        }
        // The last page opening at or before `pos`, found by BASE — so a
        // position landing in a hole between two pages resolves to the page
        // before it, with all of that page's rows below.
        let p = match self.bases.binary_search(&limit) {
            // A page opening exactly at `limit` opens after `pos`.
            Ok(i) => return self.page_first[i] as usize,
            Err(0) => return 0,
            Err(i) => i - 1,
        };
        let base_rows = self.page_first[p] as usize;
        let inside = limit - self.bases[p];
        // Whole rows of this page that fit. The page's last row is short, so it
        // only counts when the position reaches the page's full token span.
        let page_rows = self.pages[p].rows().unwrap_or(0);
        let all = if inside >= self.page_tokens[p] {
            page_rows
        } else {
            (inside / self.ratio).min(page_rows.saturating_sub(1))
        };
        base_rows + all
    }

    /// Rotate every page into the frame of the position it sits at, write the
    /// scorer's channel-blocked staging, and materialise the descriptor table.
    /// Idempotent.
    ///
    /// Both jobs are the one pass — see [`Self::placed`].
    #[cfg(feature = "cuda")]
    pub fn build_tables(&mut self, rope: &RopeTables) -> Result<()> {
        if self.keys_tbl.is_some() {
            return Ok(());
        }
        let jobs: Vec<PlacePage<'_>> = self
            .pages
            .iter()
            .zip(&self.bases)
            .map(|(p, &base)| PlacePage {
                keys: &p.keys,
                delta: base as isize - p.roped_base as isize,
            })
            .collect();
        let placement = Placement::plan(&jobs)?;
        placement.run(rope, PLACE_TILE_R)?;
        let mut ptrs: Vec<i64> = Vec::with_capacity(self.pages.len());
        for t in placement.staged() {
            ptrs.push(super::indexer::tensor_ptr(t)? as i64);
        }
        self.placed = Some(placement);
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
        rope: &RopeTables,
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
        self.build_tables(rope)?;
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
    #[cfg(feature = "cuda")]
    pub fn score_reference(
        &self,
        q: &Tensor,
        qpos: &[usize],
        n_heads: usize,
        head_dim: usize,
        rope: &RopeTables,
    ) -> Result<Vec<f32>> {
        let t = qpos.len();
        let n = self.total_rows();
        let qv = q.flatten_all()?.to_vec1::<f32>()?;
        // **Placed, like the kernel's operand.** The scorer reads each page in
        // the frame of the position it sits at, so an oracle reading the record
        // frame would be comparing two different tensors and calling the
        // difference a kernel bug. Rotated here with the HOST rope — a different
        // implementation from the kernel's, which is the whole point of an
        // oracle.
        let mut keys: Vec<f32> = Vec::with_capacity(n * head_dim);
        for (p, &base) in self.pages.iter().zip(&self.bases) {
            let rows = p.rows()?;
            let delta = base.saturating_sub(p.roped_base);
            let placed = rope
                .apply_at_positions(&p.keys.reshape((rows, 1, head_dim))?, &vec![delta; rows])?
                .reshape((rows, head_dim))?;
            keys.extend(placed.flatten_all()?.to_vec1::<f32>()?);
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

/// Which of a sequence's pages a trailing span of `tokens` tokens reaches back
/// into, and what those pages plus the live tail actually cover.
///
/// Returns `(first_page, covered)`: the seal takes pages `[first_page, N)` plus
/// the live tail, and `covered` is the tokens that set spans.
///
/// **A page is indivisible, so the walk never takes one that would reach back
/// past the span's start.** Taking a page to satisfy a few residual tokens
/// drags its whole width in, and those rows then describe tokens the span does
/// not own — which is worse than leaving the residue uncovered, because the
/// surplus rows claim positions the borrowing slot does not hold and are
/// selected against without erroring. A shortfall is at least visible: it is the
/// direction `score_rows` refuses on.
///
/// So the walk stops when the next page is wider than what is still needed, and
/// `covered` may fall short of `tokens`. It may also exceed it without any page
/// being taken, when the live tail alone already over-spans. Either way the
/// caller is told, which is why `covered` is returned rather than assumed.
///
/// **This is the fix for a measured regression.** Before it, a turn whose live
/// tail covered three fewer tokens than its K/V pulled in an entire preceding
/// 323-token page, and every projection borrowing that turn carried an index 320
/// tokens wider than the turn. Against the committed baseline on the same
/// workload that was the difference between "index 3 past" and "index 323 past"
/// on every ingest slot — and it only appeared once page cuts split a turn's
/// rows so that a walk-back was needed at all.
///
/// A free function so the boundary arithmetic is testable without a device,
/// exactly as `front_evict_count` is for the eviction ring. Every input here is
/// a token count; nothing touches a tensor.
pub fn tail_span_pages(page_widths: &[usize], tail_tokens: usize, tokens: usize) -> (usize, usize) {
    let mut need = tokens.saturating_sub(tail_tokens);
    let mut first_page = page_widths.len();
    let mut covered = tail_tokens;
    while need > 0 && first_page > 0 {
        let width = page_widths[first_page - 1];
        // Taking this page would reach back past the span's own start.
        if width > need {
            break;
        }
        first_page -= 1;
        covered += width;
        need -= width;
    }
    (first_page, covered)
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
        // A decoded page is position-free by construction: the seal normalised
        // it before writing, so `roped_base` is zero and its placement rotation
        // is exactly the base it is placed at.
        page: IndexPage::new(
            Tensor::from_vec(vals, (rows, dim), dev)?.to_dtype(DType::F32)?,
            last_cells,
        ),
        open: Tensor::from_vec(open_vals, (n_open, dim), dev)?.to_dtype(DType::F32)?,
    })
}

#[cfg(test)]
mod tail_span_tests {
    use super::tail_span_pages;

    /// A turn's rows are sealed by WIDTH — the seal walks back from the live
    /// tail over whole pages until the turn is covered. These pin what that walk
    /// covers, because **the pages it returns become the turn's stored index**,
    /// and a projection borrowing the turn pushes exactly them. If they span
    /// more tokens than the turn holds, its index claims positions the turn does
    /// not own; if fewer, part of the turn is unindexed. Either way the slot
    /// diverges from its K/V, silently below the QSA identity threshold and as a
    /// refused select above it.
    ///
    /// `covered == tokens` is therefore the contract, and every case below says
    /// whether the walk can honour it.
    /// The whole turn is still in the live tail — the common case, and the one
    /// that is always exact.
    #[test]
    fn a_turn_inside_the_live_tail_takes_no_pages() {
        assert_eq!(tail_span_pages(&[100, 200], 50, 50), (2, 50));
        // Zero-width turn: nothing to cover, nothing taken.
        assert_eq!(tail_span_pages(&[100], 50, 0), (1, 50));
        // No pages at all (a sequence that forwarded everything it holds).
        assert_eq!(tail_span_pages(&[], 40, 40), (0, 40));
    }

    /// A turn whose rows were split across page boundaries by reprojection —
    /// the case width-based sealing exists for. Exact whenever the turn begins
    /// on a page boundary.
    #[test]
    fn a_turn_spanning_whole_pages_is_exact() {
        // tail 30 + page[1]=70 == 100.
        assert_eq!(tail_span_pages(&[300, 70], 30, 100), (1, 100));
        // tail 30 + page[2]=70 + page[1]=100 == 200.
        assert_eq!(tail_span_pages(&[300, 100, 70], 30, 200), (1, 200));
        // The turn is the whole sequence: every page plus the tail.
        assert_eq!(tail_span_pages(&[300, 100, 70], 30, 500), (0, 500));
    }

    /// **The regression, and the rule that fixes it.** A turn whose live tail
    /// falls a few tokens short of its K/V must NOT drag in a whole preceding
    /// page to make up the difference.
    ///
    /// These are the live numbers. An ingest turn 323 tokens wide, whose tail
    /// covers 3 fewer, sat beside a preceding 323-token page. Taking that page
    /// made the turn's stored index span 320 tokens more than the turn — the
    /// exact difference between the committed baseline ("index 3 past") and this
    /// branch ("index 323 past") on every ingest slot, measured on the daemon.
    #[test]
    fn a_small_shortfall_never_drags_in_a_whole_page() {
        // 3 tokens still needed, a 323-token page behind: refuse it.
        let (first, covered) = tail_span_pages(&[323, 323], 320, 323);
        assert_eq!(
            first, 2,
            "the walk must not step back for a 3-token residue"
        );
        assert_eq!(covered, 320, "so the tail alone is what the turn seals");
        assert!(
            covered <= 323,
            "over-spanning is the direction that claims positions the slot does \
             not hold, and is selected against without erroring"
        );

        // One token short: same refusal.
        assert_eq!(tail_span_pages(&[323, 323], 322, 323), (2, 322));
    }

    /// **The measured failure, both halves of it, at the live numbers.**
    ///
    /// A repo_map ingest slot held a 323-token projected prefix in the index's
    /// live tail, then prefilled a 53-token turn into the same tail and decoded
    /// 2 tokens — a 55-token turn whose rows did not begin on a page boundary.
    /// Whichever way the walk resolves that, it is wrong, and this pins both so
    /// neither can come back as a "fix" for the other:
    ///
    /// - the tail alone spans 378 for a 55-token turn — over by exactly the
    ///   prefix, which is the original `index 323 past` on every borrowed turn;
    /// - closing the tail first makes one 376-token page the walk must refuse,
    ///   leaving 2 — the `index 53 short` it became.
    ///
    /// The cure is neither arithmetic: it is the page boundary at the turn's
    /// start (`apply_segments_finish`), after which the third case holds and
    /// `covered == tokens` exactly.
    #[test]
    fn a_turn_sharing_a_tail_with_its_prefix_cannot_be_sealed_either_way() {
        // (a) No boundary at all: the tail holds prefix + turn.
        let (first, covered) = tail_span_pages(&[], 323 + 53 + 2, 55);
        assert_eq!(first, 0);
        assert_eq!(
            covered, 378,
            "the tail alone over-spans by the whole prefix"
        );
        assert_eq!(covered - 55, 323, "which is the 323 measured on the daemon");

        // (b) Boundary in the wrong place: prefix and turn prefill closed
        // together into one 376-token page, leaving only the decoded tail.
        let (first, covered) = tail_span_pages(&[376], 2, 55);
        assert_eq!(
            first, 1,
            "the walk must refuse a page 376 wide for 53 needed"
        );
        assert_eq!(covered, 2, "so the turn seals almost no rows");

        // (c) Boundary at the turn's start — what the fix installs. The prefix
        // is its own page and the turn's rows are the tail, exactly.
        let (first, covered) = tail_span_pages(&[323], 55, 55);
        assert_eq!(
            (first, covered),
            (1, 55),
            "exact: no page taken, no residue"
        );

        // (d) …and it stays exact once the turn's own cuts split it further:
        // [prefix][turn prefill][reasoning] as pages, the answer as the tail.
        let (first, covered) = tail_span_pages(&[323, 53, 30], 25, 108);
        assert_eq!(
            first, 1,
            "takes the turn's own pages and stops at the prefix"
        );
        assert_eq!(covered, 108, "53 + 30 + 25 — the turn, exactly");
    }

    /// **The pages are right; the width they are asked for is short.**
    ///
    /// A real layout, copied from a daemon run: an ingest turn whose index ends
    /// `[…, 988, 5]` over a live tail of 69, sealed against a `turn_token_count`
    /// of 1057. The walk covers 74 and refuses the 988 — and `988 + 69 == 1057`
    /// exactly, which is what makes the small page look like an interloper.
    ///
    /// The discriminating fact is the second half: hand the SAME page list the
    /// width that includes that page and the walk lands exact, with no residue
    /// and no page refused. So the cut positions, the widths and the walk are
    /// all self-consistent, and the only quantity that does not fit is the
    /// turn's own token count. Whatever produces the 5-token page produces real
    /// rows for real forwarded tokens; the count the seal is given omits them.
    ///
    /// This held on 44 of 44 sealed turns in one run, with the small page
    /// varying (5, 6, 7) — so it tracks content, and is not a fixed marker.
    #[test]
    fn the_same_pages_seal_exactly_once_the_width_includes_every_one_of_them() {
        const PAGES: &[usize] = &[
            3, 4, 59, 47, 9, 14, 108, 49, 234, 4, 284, 398, 4, 2, 53, 2, 142, 2, 988, 5,
        ];
        const TAIL: usize = 69;

        // The count the seal was actually given — the K/V chunk sum.
        let (first, covered) = tail_span_pages(PAGES, TAIL, 1057);
        assert_eq!(first, PAGES.len() - 1, "only the 5-token page is taken");
        assert_eq!(covered, TAIL + 5, "matching the run's covered=74");
        assert_eq!(
            988 + TAIL,
            1057,
            "the refused page plus the tail IS the count"
        );

        // The same pages, asked for the width that also covers the 5.
        let (first, covered) = tail_span_pages(PAGES, TAIL, 988 + 5 + TAIL);
        assert_eq!(
            (first, covered),
            (PAGES.len() - 2, 1062),
            "exact: both of the turn's own pages taken, nothing refused"
        );
    }

    /// **A turn whose user half was prefilled during projection needs a
    /// boundary there, not only at the view carve.**
    ///
    /// The numbers are a daemon run's, verbatim: `turn_token_count=324`,
    /// `reasoning=(296, 3)`, live tail 28, sealed `widths=[7, 28]`, `covered=35`.
    /// The turn is a 289-token user message, then a 7-token assistant header
    /// forwarded on the view, then 28 decoded tokens.
    ///
    /// With a boundary only at the carve, the user half has already been
    /// prefilled onto the parent and is closed into the prefix's page — so the
    /// turn's own "prefill" page is just the 7-token header, and the walk refuses
    /// the page holding its user message. With a boundary before the deferred
    /// user prefill as well, the same turn seals exactly.
    #[test]
    fn a_prefilled_user_half_needs_its_own_boundary_or_it_seals_with_the_prefix() {
        const PREFIX: usize = 1000;
        const USER: usize = 289;
        const HEADER: usize = 7;
        const DECODED: usize = 28;
        const TURN: usize = USER + HEADER + DECODED;
        assert_eq!(TURN, 324, "the turn the run sealed");

        // Only the carve closes: the user half is inside the prefix's page.
        let (first, covered) = tail_span_pages(&[PREFIX + USER, HEADER], DECODED, TURN);
        assert_eq!(first, 1, "the page holding the user message is refused");
        assert_eq!(
            covered,
            HEADER + DECODED,
            "reproducing the run's covered=35 exactly"
        );

        // Both boundaries close: prefix, user half, header — then the tail.
        let (first, covered) = tail_span_pages(&[PREFIX, USER, HEADER], DECODED, TURN);
        assert_eq!(
            first, 1,
            "stops at the prefix, taking both of the turn's pages"
        );
        assert_eq!(covered, TURN, "the turn, exactly");
    }

    /// The residue is bounded by the page that was refused, never by the whole
    /// sequence: the walk still takes every page it can fully justify.
    #[test]
    fn the_walk_takes_every_page_it_can_justify_before_stopping() {
        // Needs 175: takes 100 and 70 (=170), then refuses the 300.
        let (first, covered) = tail_span_pages(&[300, 100, 70], 30, 205);
        assert_eq!(first, 1);
        assert_eq!(covered, 30 + 70 + 100);
        assert!(covered <= 205);
    }

    /// The mirror: a live tail WIDER than the turn. No page is taken, but the
    /// tail alone already over-spans — width-based sealing cannot be exact here
    /// either, and the caller is told rather than left to assume.
    #[test]
    fn a_tail_wider_than_the_turn_over_spans_without_walking() {
        assert_eq!(tail_span_pages(&[100], 60, 50), (1, 60));
        assert_eq!(tail_span_pages(&[], 60, 50), (0, 60));
    }

    /// Asking for more than the sequence holds takes every page and stops,
    /// rather than looping or panicking.
    #[test]
    fn a_span_past_the_whole_sequence_stops_at_the_first_page() {
        assert_eq!(tail_span_pages(&[10, 20], 5, 10_000), (0, 35));
    }

    /// Zero-width pages are taken freely — they cost nothing and cover nothing —
    /// and never stall the walk. The refusal still applies to the real page
    /// behind them: 10 tokens of residue does not justify a 50-token page, so
    /// the walk stops there having taken only the empties.
    #[test]
    fn zero_width_pages_do_not_stall_the_walk() {
        assert_eq!(tail_span_pages(&[50, 0, 0], 10, 20), (1, 10));
        // Nothing needed at all: no page is taken, empty or otherwise.
        assert_eq!(tail_span_pages(&[0, 0], 10, 10), (2, 10));
    }

    /// **The invariant, over the whole input space.** For any page layout, tail
    /// and span, the walk must never reach back past the span's start — the one
    /// direction that silently corrupts a borrowing slot. The only permitted
    /// over-span is a live tail that already exceeds the request before any page
    /// is considered, which no choice of pages can undo.
    #[test]
    fn the_walk_never_reaches_back_past_the_span_start() {
        let layouts: [&[usize]; 6] = [
            &[],
            &[1],
            &[323, 323],
            &[300, 100, 70],
            &[7, 7, 7, 7],
            &[1000, 1, 1000, 1],
        ];
        for pages in layouts {
            for tail in 0..40usize {
                for tokens in 0..80usize {
                    let (first, covered) = tail_span_pages(pages, tail, tokens);
                    assert!(first <= pages.len());
                    let taken: usize = pages[first..].iter().sum();
                    assert_eq!(
                        covered,
                        tail + taken,
                        "covered must be the tail plus exactly the pages taken"
                    );
                    if tail <= tokens {
                        assert!(
                            covered <= tokens,
                            "pages={pages:?} tail={tail} tokens={tokens} over-spanned \
                             to {covered}"
                        );
                    } else {
                        assert_eq!(
                            covered, tail,
                            "an over-wide tail must not additionally take pages"
                        );
                    }
                }
            }
        }
    }
}
