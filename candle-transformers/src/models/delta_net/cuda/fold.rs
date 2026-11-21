//! Applies the kept timeline rows of a step to states and conv tails in place,
//! for every sequence and layer, in one launch.
//!
//! A fold is issued after a forward, when no wave span is standing. Its pointer
//! table is held ground, and it is dead once the launch is behind it. Every entry
//! is checked before anything launches, so a refusal writes nothing. Entries
//! beyond the grid's z extent are launched in further passes. The tensor-based
//! form, `delta_net_fold_states`, serves the single-sequence entry points and the
//! kernel tests.

use super::*;

/// Whether the fold kernel serves `dims`: the fused head width, and a conv
/// window the tail's registers hold.
pub fn fold_runs_fused(dims: &DeltaNetDims) -> bool {
    dims.head_dim == DELTA_NET_PREFILL_DIM
        && (2..=DELTA_NET_PREFILL_CONV_MAX).contains(&dims.conv_kernel)
}

/// One sequence's layer to fold: its conv tail, state and timeline addresses,
/// and the rows to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldEntry {
    pub tail: u64,
    pub state: u64,
    pub timeline: u64,
    pub m: usize,
}

/// Apply every entry's first `m` timeline rows to its state and conv tail, in
/// place, in ONE launch however many sequences and layers it covers — up to the
/// grid's z extent, [`DELTA_NET_MAX_SPANS`] entries, a launch each beyond it
/// (`delta_net_fold_kernel.cuh`).
///
/// The table — three pointers and a count per entry, one `[4, n]` upload — goes
/// on held ground: a fold is issued after a forward, where no wave span stands,
/// and its table is dead once the launch is behind it.
pub fn delta_net_fold(entries: &[FoldEntry], dims: &DeltaNetDims, device: &Device) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    if !fold_runs_fused(dims) {
        candle::bail!(
            "delta_net cuda: the fold kernel runs at d == {DELTA_NET_PREFILL_DIM} with a conv \
             window of 2..={DELTA_NET_PREFILL_CONV_MAX}, got d {} and window {}",
            dims.head_dim,
            dims.conv_kernel
        );
    }
    let Device::Cuda(dev) = device else {
        candle::bail!("delta_net cuda: the fold kernel runs on a CUDA device");
    };
    // **Eager, in issue order behind whatever this thread has recorded.** A fold
    // is issued at a forward's commit and at a step's reconcile, both inside a
    // wave capture, and its table is held ground: giving the range back fences
    // it on the stream, which a recording stream cannot take. Declared first so
    // it is dropped last — the table goes back while the stream is still eager.
    let _eager = device.eager()?;
    // Every entry checked before anything launches, so a refusal writes nothing.
    for e in entries {
        if e.m == 0 || e.m > TIMELINE_ROWS {
            candle::bail!(
                "delta_net cuda: a fold of {} rows from a {TIMELINE_ROWS}-row timeline",
                e.m
            );
        }
        if !e.state.is_multiple_of(16) || !e.timeline.is_multiple_of(16) {
            candle::bail!(
                "delta_net cuda: a fold entry's state or timeline is not 16-byte aligned"
            );
        }
    }
    // One launch per grid's worth: an entry is a block column of the grid's z
    // extent. The launcher refuses only by geometry, which every chunk shares,
    // so a refusal comes on the first launch, before anything is written.
    let stream = dev.cuda_stream();
    for chunk in entries.chunks(DELTA_NET_MAX_SPANS) {
        let n = chunk.len();
        let mut table: Vec<i64> = Vec::with_capacity(FOLD_TABLE_ROWS * n);
        table.extend(chunk.iter().map(|e| e.tail as i64));
        table.extend(chunk.iter().map(|e| e.state as i64));
        table.extend(chunk.iter().map(|e| e.timeline as i64));
        table.extend(chunk.iter().map(|e| e.m as i64));
        let table_t = held_empty((FOLD_TABLE_ROWS, n), DType::I64, device)?;
        upload_into(&table_t, &table)?;
        let status = unsafe {
            run_delta_net_fold_f32(
                typed_ptr(&table_t, DType::I64, "fold table")? as *const i64,
                n as i32,
                dims.n_v_heads as i32,
                dims.n_k_heads as i32,
                dims.conv_dim() as i32,
                dims.conv_kernel as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        if status != 0 {
            candle::bail!(
                "delta_net cuda: the fold refused {n} entries over h_v {} / h_k {} and wrote \
                 nothing",
                dims.n_v_heads,
                dims.n_k_heads
            );
        }
    }
    Ok(())
}

/// [`delta_net_fold`] over states and timelines held as tensors — the
/// single-sequence entry points and the kernel tests.
pub fn delta_net_fold_states(
    jobs: &[(&DeltaNetState, &Tensor, usize)],
    dims: &DeltaNetDims,
) -> Result<()> {
    let Some((first, _, _)) = jobs.first() else {
        return Ok(());
    };
    let lay = TimelineLayout::of(dims);
    let entries = jobs
        .iter()
        .map(|(st, tl, m)| {
            if tl.dims1()? != lay.elems() {
                candle::bail!(
                    "delta_net cuda: a {:?} timeline for a geometry of {} elements",
                    tl.dims(),
                    lay.elems()
                );
            }
            Ok(FoldEntry {
                tail: f32_ptr(&st.conv_tail, "conv tail")?,
                state: f32_ptr(&st.s, "state")?,
                timeline: f32_ptr(tl, "timeline")?,
                m: *m,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let device = first.s.device().clone();
    delta_net_fold(&entries, dims, &device)
}
