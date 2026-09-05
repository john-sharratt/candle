//! The block-sparse selection an attention layer reads through.
//!
//! This is the *carrier*: the device-side pair of tensors the paged attention
//! kernels take, plus the pointer resolution that hands them over. The
//! semantics — which cells a query attends, how an entry is packed, and why —
//! belong to the architecture that has them, and are stated once in
//! [`crate::models::qwen4exp::qsa_select`] with the kernel-side mirror in
//! `candle-kernels/src/qsa_select.cuh`.
//!
//! A layer with no selection passes `None` and the kernels read the whole
//! causal prefix: that is not a disabled feature, it is the absence of one —
//! the state every attention layer outside Qwen3.8-Flash-Next's twelve is in.

use candle::{DType, Result, Tensor};

/// One layer's selection over one wave's query rows.
///
/// Rows are indexed the way the kernel indexes queries: by **slot** on the
/// decode path (one query per slot), and by **packed query row** on the
/// prefill path (`q_start + token`).
#[derive(Debug, Clone)]
pub struct QsaSelection {
    /// `[n_rows, stride]` U32 — each row's packed entries, ascending by block,
    /// padded past its count.
    entries: Tensor,
    /// `[n_rows]` U32 — entries per row, or the dense marker.
    cnt: Tensor,
    /// The page layout, as `(pages, windows)`:
    ///
    /// - `pages` — `[n_pages, 2]` U32, `{tokens_before, blocks_before}` per
    ///   page, ascending, concatenated over the launch's sequences.
    /// - `windows` — `[n_rows, 2]` U32, `{offset, count}` into `pages` for each
    ///   row's sequence.
    ///
    /// **What makes a block's position span knowable when the prefix arrived as
    /// separately sealed pieces.** Uniformly, block `b` covers
    /// `[b·ratio, (b+1)·ratio)` and the kernels compute that inline. A piece
    /// ends where its tokens ended, so its last block is short and every block
    /// after it is displaced — the map stops being arithmetic and becomes a
    /// walk over these prefixes.
    ///
    /// `None` for a sequence that forwarded everything it holds, which is every
    /// sequence outside the projection path; the kernels then take the inline
    /// arithmetic and never touch this.
    ///
    /// **One `Option` over the pair, not one each.** The two tables are only
    /// meaningful together — pages without windows says nothing about which
    /// pages a row reads, windows without pages index nothing — and a
    /// half-populated pair would read as "no pages" at the launch, silently
    /// restoring the inline arithmetic the tables exist to replace. That is a
    /// wrong position with no fault, so the pairing is made unrepresentable
    /// rather than checked.
    pages: Option<(Tensor, Tensor)>,
    /// Cells per index block (the layer's `compress_ratio`).
    ratio: usize,
}

impl QsaSelection {
    /// Validate the shapes the kernels assume and take ownership of the views.
    pub fn new(entries: Tensor, cnt: Tensor, ratio: usize) -> Result<Self> {
        let (rows, stride) = entries.dims2()?;
        if cnt.dims1()? != rows {
            candle::bail!(
                "qsa selection: {} entry rows against {} counts",
                rows,
                cnt.dims1()?
            );
        }
        if entries.dtype() != DType::U32 || cnt.dtype() != DType::U32 {
            candle::bail!(
                "qsa selection: entries/cnt must be U32 (got {:?}/{:?})",
                entries.dtype(),
                cnt.dtype()
            );
        }
        if ratio == 0 {
            candle::bail!("qsa selection: ratio must be nonzero");
        }
        if stride == 0 {
            candle::bail!("qsa selection: entry stride must be nonzero");
        }
        Ok(Self {
            entries,
            cnt,
            pages: None,
            ratio,
        })
    }

    /// Attach the page layout for sequences whose prefix arrived as sealed
    /// pieces — see [`Self::pages`].
    pub fn with_pages(mut self, pages: Tensor, page_win: Tensor) -> Result<Self> {
        let (n_pages, pw) = pages.dims2()?;
        let (n_rows, ww) = page_win.dims2()?;
        if pw != 2 || ww != 2 {
            candle::bail!("qsa selection: page tables must be [_, 2] (got {pw} / {ww} wide)");
        }
        if n_rows != self.rows() {
            candle::bail!(
                "qsa selection: {n_rows} page windows against {} entry rows",
                self.rows()
            );
        }
        if n_pages == 0 {
            candle::bail!("qsa selection: a page table with no pages describes nothing");
        }
        if pages.dtype() != DType::U32 || page_win.dtype() != DType::U32 {
            candle::bail!("qsa selection: page tables must be U32");
        }
        self.pages = Some((pages, page_win));
        Ok(self)
    }

    /// Query rows this selection covers.
    pub fn rows(&self) -> usize {
        self.entries.dim(0).unwrap_or(0)
    }

    /// Row pitch of `entries`, in u32s — the kernels' `sel_stride`.
    pub fn stride(&self) -> usize {
        self.entries.dim(1).unwrap_or(0)
    }

    /// Cells per index block — the kernels' `sel_ratio`.
    pub fn ratio(&self) -> usize {
        self.ratio
    }

    pub fn entries(&self) -> &Tensor {
        &self.entries
    }

    pub fn cnt(&self) -> &Tensor {
        &self.cnt
    }

    /// Resolve the kernel arguments and hold the storages for the whole of `f`.
    ///
    /// One definition for every launch site: `None` hands the kernel a null
    /// pointer pair, which its `qsa_active` reads as "no restriction". The
    /// storage guards live across `f`, so the launch inside it cannot outlive
    /// the buffers it was given.
    #[cfg(feature = "cuda")]
    pub fn with_kernel_args<R>(
        sel: Option<&Self>,
        stream: &candle::cuda_backend::cudarc::driver::CudaStream,
        f: impl FnOnce(*const u32, *const u32, *const u32, *const u32, i32, i32) -> R,
    ) -> Result<R> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;

        let Some(sel) = sel else {
            return Ok(f(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                1,
            ));
        };
        let (e_s, e_l) = sel.entries.storage_and_layout();
        let (c_s, c_l) = sel.cnt.storage_and_layout();
        let e_slice = match &*e_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: entries must be CUDA"),
        }
        .slice(e_l.start_offset()..);
        let c_slice = match &*c_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("qsa selection: cnt must be CUDA"),
        }
        .slice(c_l.start_offset()..);
        // Both rows must be readable as a plain pitched table — the kernels
        // index `entries[row * stride + i]` with no layout metadata.
        if !e_l.is_contiguous() || !c_l.is_contiguous() {
            candle::bail!("qsa selection: entries/cnt must be contiguous");
        }
        let (e_ptr, _e_g) = e_slice.device_ptr(stream);
        let (c_ptr, _c_g) = c_slice.device_ptr(stream);

        // The page tables, when this launch's sequences have one. Bound in this
        // scope so their storage guards live exactly as long as the entry
        // tables' do — the launch inside `f` reads all four.
        let page_storage = sel
            .pages
            .as_ref()
            .map(|(p, w)| (p.storage_and_layout(), w.storage_and_layout()));
        let page_slices = match &page_storage {
            Some(((p_s, p_l), (w_s, w_l))) => {
                if !p_l.is_contiguous() || !w_l.is_contiguous() {
                    candle::bail!("qsa selection: page tables must be contiguous");
                }
                let p = match &**p_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                    _ => candle::bail!("qsa selection: page tables must be CUDA"),
                }
                .slice(p_l.start_offset()..);
                let w = match &**w_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                    _ => candle::bail!("qsa selection: page tables must be CUDA"),
                }
                .slice(w_l.start_offset()..);
                Some((p, w))
            }
            None => None,
        };
        let mut pages_ptr: *const u32 = std::ptr::null();
        let mut win_ptr: *const u32 = std::ptr::null();
        let _page_guards = page_slices.as_ref().map(|(p, w)| {
            let (pp, p_g) = p.device_ptr(stream);
            let (wp, w_g) = w.device_ptr(stream);
            pages_ptr = pp as *const u32;
            win_ptr = wp as *const u32;
            (p_g, w_g)
        });

        Ok(f(
            e_ptr as *const u32,
            c_ptr as *const u32,
            pages_ptr,
            win_ptr,
            sel.stride() as i32,
            sel.ratio as i32,
        ))
    }

    /// The rows `[start, start + len)` of this selection, as a selection.
    ///
    /// A wave hands attention its decode and prefill groups separately, over
    /// one packed row order; each group takes its own window of the same
    /// tables. The narrows are views — no copy (hot-path invariant 2).
    pub fn rows_slice(&self, start: usize, len: usize) -> Result<Self> {
        Ok(Self {
            entries: self.entries.narrow(0, start, len)?,
            cnt: self.cnt.narrow(0, start, len)?,
            // The windows are per row and narrow with them; the page table is
            // indexed BY those windows and stays whole, so the offsets they
            // carry keep pointing at the right pages.
            pages: match &self.pages {
                Some((p, w)) => Some((p.clone(), w.narrow(0, start, len)?)),
                None => None,
            },
            ratio: self.ratio,
        })
    }
}
