//! Projects a wave's index keys and queries, one GEMM over all of its rows.
//!
//! Projecting per sequence would put a small GEMM per sequence per layer on the
//! decode path, where the launch count is the cost. Both projections therefore
//! cover the whole wave, and each cache takes its own row window. Queries are
//! normed and then rotated at each row's own position and rung, in the order the
//! reference uses, from the same table the scorer rotates keys with.

use super::*;

/// The wave's raw index keys: one GEMM over every row, not one per sequence.
///
/// `h` is the layer's `[rows, hidden]` block input — the same rows the
/// attention projections read, which is what the reference projects its index
/// keys from. The per-sequence caches then take their own row window
/// ([`IndexCache::append`]); splitting the projection instead would put one
/// small GEMM per sequence per layer on the decode path, where launches are
/// the wall.
pub fn project_keys(h: &Tensor, w: &IndexerWeights) -> Result<Tensor> {
    rows_matmul_t(h, &w.k_proj)
}

/// The wave's indexer queries — projected, normed, and rotated at each row's
/// own absolute position, in one pass over every row.
///
/// The reference's order (`qsa_selection_mask`): project, RMS-norm with
/// `q_norm`, then rotate — from the same factored table the scorer rotates the
/// keys from, so the two sides of every dot product share one set of
/// frequencies. Each row rotates at its sequence's rung (`rungs`), and takes
/// that rung's `m²` as the attention's queries do (§12).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn project_queries(
    h: &Tensor,
    w: &IndexerWeights,
    cfg: &IndexerConfig,
    rope: &FactoredRope,
    positions: &[usize],
    rungs: RowRungs<'_>,
    rms_eps: f64,
    // The open layer phase, for the rotation's position and rung tables.
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    let rows = h.dim(0)?;
    let q = rows_matmul_t(h, &w.q_proj)?.reshape((rows * cfg.n_heads, cfg.head_dim))?;
    rotate_rows(
        RowSource::Normed {
            rows: &q,
            weight: &w.q_norm,
            eps: rms_eps,
        },
        rope,
        cfg.n_heads,
        RowPositions::PerGroup(positions),
        rungs,
        RotSide::Query,
        ticket,
    )?
    .reshape((rows, cfg.n_heads, cfg.head_dim))
}

/// `out[row_base + r, j] = Σ_h relu(raw[r · h + i, j])`, in one launch.
///
/// `raw` is the scoring matmul's `[rows · h, m]` output, whose row axis carries
/// (row, head) in that order — so as a `[rows, h, m]` view its strides are
/// `(h · m, m, 1)`, which is exactly what the kernel reads through. `out` is the
/// wave's `[total_rows, out_stride]` score buffer; columns past this span's `m`
/// are left as allocated, and the top-k never reads them because each row's scan
/// is bounded by its own candidate count.
#[allow(clippy::too_many_arguments)]
pub(super) fn fold_heads_into(
    raw: &Tensor,
    rows: usize,
    h: usize,
    m: usize,
    out: &Tensor,
    out_stride: usize,
    row_base: usize,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_kernels::simple::indexer_score::run_indexer_score_reduce;

    let candle::Device::Cuda(dev) = raw.device() else {
        candle::bail!("qsa selection runs on CUDA");
    };
    if m > out_stride {
        candle::bail!("qsa selection: {m} candidates into a {out_stride}-wide score row");
    }
    let stream = dev.cuda_stream();
    let (r_s, r_l) = raw.storage_and_layout();
    let r_slice = match &*r_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qsa selection: scores must be CUDA"),
    }
    .slice(r_l.start_offset()..);
    let (o_s, o_l) = out.storage_and_layout();
    let o_slice = match &*o_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("qsa selection: fold output must be CUDA"),
    }
    .slice(o_l.start_offset() + row_base * out_stride..);
    let (r_ptr, _rg) = r_slice.device_ptr(&stream);
    let (o_ptr, _og) = o_slice.device_ptr(&stream);
    let raw_stream = stream.cu_stream() as *mut c_void;
    candle::set_kernel_breadcrumb("run_indexer_score_reduce", file!(), line!());
    unsafe {
        run_indexer_score_reduce(
            r_ptr as *const f32,
            std::ptr::null(),
            std::ptr::null(),
            o_ptr as *mut f32,
            rows as i32,
            h as i32,
            m as i32,
            (h * m) as i64,
            m as i64,
            1,
            0,
            0,
            0,
            out_stride as i64,
            raw_stream,
        );
    }
    Ok(())
}
