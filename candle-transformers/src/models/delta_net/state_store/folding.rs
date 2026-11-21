//! Folds the pending timeline rows of recurrent layers into their state, in one
//! launch for a whole set of stores.
//!
//! A short span's rows wait in its timeline until a fold applies the rows a step
//! keeps, so an undo can still drop them until then. Folds run between waves,
//! since a store with a wave open may still be writing its timelines. A fold that
//! is refused writes nothing, which lets the caller roll the waves back and leave
//! every store as it entered.

use super::*;

/// What a step's waves left pending, so the step can be kept or taken back
/// whole — see [`RecurrentStateStore::begin_step`].
#[derive(Debug)]
pub(super) struct StepRecord {
    /// Per slot: whether it held rows when the step began. They are not the
    /// step's — an undo leaves them — and no span of the step ran that layer,
    /// because a pending layer refuses one.
    pub(super) entered_pending: Vec<bool>,
    /// A commit inside the step owed this store a fold — it is outside the
    /// armed verify cohort — which the step's end runs instead.
    pub(super) fold_owed: bool,
    /// Something an undo cannot take back happened inside the step: a long
    /// span's exchange, a fold, or the live state written outside a wave.
    pub(super) broken: bool,
}

/// Fold every pending layer of every store in `jobs`, each store keeping the
/// first `kept` rows of its timelines — or every row, for `None` — and end
/// pending on all of them.
///
/// **One launch for the whole set** on CUDA (`delta_net_fold`): a step folds
/// every sequence it ran, every recurrent layer of each, together. A store
/// keeping `Some(0)` rows folds nothing and simply drops its timelines — a span
/// that committed nothing.
///
/// Between waves: a store with a wave open refuses, because its timelines may
/// still be being written. A forward's own fold goes through [`commit_stores`].
pub fn fold_stores(jobs: &mut [(&mut RecurrentStateStore, Option<usize>)]) -> Result<()> {
    if jobs.iter().any(|(s, _)| s.open) {
        candle::bail!(
            "recurrent store: a fold with a wave open — the wave may still be writing the \
             timelines it would apply"
        );
    }
    fold_pending(jobs)
}

/// Close a successful wave over `stores`: fold every pending row of each store
/// marked `true` — every short span of a sequence outside the armed verify
/// cohort, in ONE launch over all of them — then commit every store's wave.
///
/// **The fold runs first, with the waves still open**, because it is the one
/// step of the close that can fail: a fold that is refused writes nothing, so
/// the caller rolls the waves back and the stores stand exactly as they
/// entered. Once it has run, the commits are infallible host bookkeeping, so a
/// wave is never reported failed over state it already moved.
///
/// **A store inside a step folds nothing here** ([`RecurrentStateStore::begin_step`]):
/// its rows stay pending, and the fold it is owed runs at [`end_steps`], so an
/// undo can still drop them.
pub fn commit_stores(stores: &mut [(&mut RecurrentStateStore, bool)]) -> Result<()> {
    let mut folds: Vec<(&mut RecurrentStateStore, Option<usize>)> = Vec::new();
    for (store, fold) in stores.iter_mut() {
        if !*fold || !store.has_pending() {
            continue;
        }
        match store.step.as_mut() {
            Some(step) => step.fold_owed = true,
            None => folds.push((&mut **store, None)),
        }
    }
    fold_pending(&mut folds)?;
    for (store, _) in stores.iter_mut() {
        store.commit_wave();
    }
    Ok(())
}

/// Close the step on every store in `stores` and keep its waves: each store a
/// commit inside the step owed a fold folds every pending row, in ONE launch
/// over all of them. A store whose rows belong to the armed verify cohort was
/// owed none, and its rows stay pending for the reconcile.
///
/// Between waves, like [`fold_stores`]. A refused fold writes nothing and the
/// rows stay pending; the steps are closed either way.
pub fn end_steps(stores: &mut [&mut RecurrentStateStore]) -> Result<()> {
    if stores.iter().any(|s| s.open) {
        candle::bail!("recurrent store: a step closed with a wave open");
    }
    let mut folds: Vec<(&mut RecurrentStateStore, Option<usize>)> = Vec::new();
    for store in stores.iter_mut() {
        let owed = store.step.take().is_some_and(|s| s.fold_owed);
        if owed && store.has_pending() {
            folds.push((&mut **store, None));
        }
    }
    fold_pending(&mut folds)
}

/// [`fold_stores`]' body, for a caller that knows no wave is still writing.
/// A fold moves the live state, so a store inside a step can no longer undo it.
pub(super) fn fold_pending(jobs: &mut [(&mut RecurrentStateStore, Option<usize>)]) -> Result<()> {
    for (store, _) in jobs.iter_mut() {
        store.break_step();
    }
    let Some((first, _)) = jobs.first() else {
        return Ok(());
    };
    let dims = first.dims;
    #[cfg(feature = "cuda")]
    let fused = matches!(first.device, Device::Cuda(_))
        && fold_runs_fused(&dims)
        && jobs
            .iter()
            .all(|(s, _)| s.slots.iter().all(|l| l.live.held.is_some()));
    #[cfg(feature = "cuda")]
    if fused {
        let mut entries: Vec<FoldEntry> = Vec::new();
        for (store, kept) in jobs.iter() {
            if store.dims != dims {
                candle::bail!("recurrent store: one fold across stores of different geometry");
            }
            entries.extend(store.fold_entries(*kept));
        }
        let device = first.device.clone();
        delta_net_fold(&entries, &dims, &device)?;
        for (store, _) in jobs.iter_mut() {
            store.discard_pending();
            store.assert_state("gdn.fold.s.L", "gdn.fold.tail.L");
        }
        return Ok(());
    }
    for (store, kept) in jobs.iter_mut() {
        let dims = store.dims;
        for slot in store.slots.iter_mut() {
            let Some(p) = slot.pending.take() else {
                continue;
            };
            let m = kept.map_or(p.len, |k| k.min(p.len));
            if m > 0 {
                fold_by_ops(&mut slot.live.state, &slot.live.timeline, m, &dims)?;
            }
        }
        // The state a fold lands is a state no wave computed directly — the
        // exit of the span that wrote the timeline — so it is checked here.
        store.assert_state("gdn.fold.s.L", "gdn.fold.tail.L");
    }
    Ok(())
}

/// Zero one layer's state and conv tail in its slot — the live half, which is
/// genuinely READ at zero, being the sequence-start state (hot-path invariant
/// 6's read-before-write case). The timeline is not touched: nothing reads it
/// before a span writes it.
///
/// Needed because a slot is recycled: it last held some other sequence's state.
#[cfg(feature = "cuda")]
pub(super) fn zero_state(dims: &DeltaNetDims, device: &Device, slot: &ArenaSlot) -> Result<()> {
    slot.zero(state_block(dims).state_end, device)
}

/// Copy one state's two buffers into another's, device to device.
///
/// The fork path's replacement for `DeltaNetState::snapshot`, which allocates.
/// Here the destination already exists — it is a view into the child's own
/// state-arena slot — so the copy writes into it rather than producing a new buffer
/// somewhere the reservation does not cover.
#[cfg(feature = "cuda")]
pub(super) fn copy_state_into(
    device: &Device,
    src: &DeltaNetState,
    dst: &DeltaNetState,
) -> Result<()> {
    let Device::Cuda(cuda) = device else {
        candle::bail!("copy_state_into: expected a CUDA device");
    };
    for (s, d) in [(&src.s, &dst.s), (&src.conv_tail, &dst.conv_tail)] {
        let bytes = s.elem_count() * s.dtype().size_in_bytes();
        let src_ptr = tensor_device_ptr(cuda, s)?;
        let dst_ptr = tensor_device_ptr(cuda, d)?;
        // On the device's stream, behind the writes that produced the source
        // and ahead of every launch that reads the copy.
        let stream = cuda.cuda_stream();
        // SAFETY: both ranges are `bytes` long, live, and disjoint — the
        // destination belongs to a store being built, which nothing else has
        // yet seen.
        unsafe {
            memcpy_dtod_async(dst_ptr, src_ptr, bytes, stream.cu_stream())
                .map_err(|e| Error::Msg(format!("recurrent fork copy: {e}")))?;
        }
    }
    Ok(())
}

/// Base device address of a contiguous CUDA tensor.
#[cfg(feature = "cuda")]
pub(super) fn tensor_device_ptr(cuda: &CudaDevice, t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        candle::bail!("recurrent state buffers are contiguous by construction");
    }
    let Storage::Cuda(c) = &*storage else {
        candle::bail!("recurrent state: expected CUDA storage");
    };
    let stream = cuda.cuda_stream();
    let base = c.slice.device_ptr(&stream);
    Ok(base + (layout.start_offset() * t.dtype().size_in_bytes()) as u64)
}

/// The slot of the store's layer `layer_index`, or the error naming it.
pub(super) fn no_slot(layer_index: usize) -> Error {
    Error::Msg(format!(
        "recurrent store: layer {layer_index} holds no recurrent state"
    ))
}
