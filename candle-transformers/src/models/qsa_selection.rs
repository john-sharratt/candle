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
            ratio,
        })
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
        f: impl FnOnce(*const u32, *const u32, i32, i32) -> R,
    ) -> Result<R> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;

        let Some(sel) = sel else {
            return Ok(f(std::ptr::null(), std::ptr::null(), 0, 1));
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
        Ok(f(
            e_ptr as *const u32,
            c_ptr as *const u32,
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
            ratio: self.ratio,
        })
    }
}
