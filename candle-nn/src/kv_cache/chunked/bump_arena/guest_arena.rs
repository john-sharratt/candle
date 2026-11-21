//! The arena that recycles a guest pipeline's stage scratch on its own ground.
//!
//! A guest's drain runs between waves with the tier handed back, so it cannot
//! use the wave tier. It installs an arena over ground its own claimed region
//! holds, and each stage opens a generation on that arena that rewinds when the
//! stage closes, so the next stage reuses the same bytes. A value a later stage
//! needs is copied out first. The arena is removed before the ground goes back
//! to the KV side, so a stale ticket cannot carve from regions that have become
//! attention state again.

use super::*;

/// The guest arena open on each device, while a drain is running.
///
/// Empty in steady state. A drain installs one over its own claimed ground and
/// removes it before that ground goes back to the KV side — so an entry here is
/// exactly as long-lived as the guest, and a ticket that outlives it resolves
/// to nothing rather than to whatever the KV side put there next.
pub(super) fn guest_domains() -> &'static Mutex<HashMap<usize, BumpArena>> {
    static DOMAINS: OnceLock<Mutex<HashMap<usize, BumpArena>>> = OnceLock::new();
    DOMAINS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether any guest arena is open, readable without taking the registry lock.
///
/// [`guest_stage`] wraps model code that is not a guest's — `ResnetBlock2D` is
/// shared with the U-Net and every stable-diffusion pipeline in the workspace —
/// so the common case by far is "no arena, run `f`". Reaching that answer
/// through [`begin_guest`] costs a process-global mutex and an allocated error
/// string per call, per resnet, per forward, for callers that will never have an
/// arena at all. This is the cheap negative: `Relaxed` is sufficient because a
/// drain installs its arena long before any stage runs and removes it long
/// after, so no stage can race the transition.
pub(super) static GUEST_ARENA_OPEN: AtomicBool = AtomicBool::new(false);

/// Put a guest arena over `base..base + capacity` for the duration of a drain.
///
/// `base` must name ground the caller holds for at least as long as the arena —
/// in practice a placement out of the guest's own `GuestGround`, which the drain
/// keeps until it drops the model. Nothing here checks that, for the same reason
/// nothing else in this module can: the address is untyped span, and the only
/// party that knows whose it is is the one that claimed it.
///
/// Replaces any arena already installed for the device. A drain is serialised
/// against every other drain by the scheduler thread, so the previous one has
/// finished by the time this runs; replacing rather than refusing means a drain
/// that died without closing cannot wedge every later one.
/// Register this module's resolver for wave tickets — idempotent, and needed by
/// anything that carves for a ticket, the guest heap included.
pub(in super::super) fn install_resolver() {
    candle::cuda_backend::wave_provenance::install_wave_allocator(resolve_wave_alloc);
}

pub fn open_guest_arena(stream: &Arc<CudaStream>, base: u64, capacity: usize) {
    install_resolver();
    let ordinal = stream.context().ordinal();
    let arena = BumpArena::over(stream, "guest", base, capacity, Reclaim::Fence);
    let mut map = guest_domains().lock().unwrap();
    map.insert(ordinal, arena);
    GUEST_ARENA_OPEN.store(true, Ordering::Relaxed);
}

/// Take the guest arena away, so no later ticket can resolve into that ground.
///
/// **Called before the guest's ground goes back**, never after: once the regions
/// return to the KV side they are attention state again, and a stale ticket
/// carving from them would hand a diffusion model's scratch the same addresses.
pub fn close_guest_arena(stream: &Arc<CudaStream>) {
    let ordinal = stream.context().ordinal();
    let mut map = guest_domains().lock().unwrap();
    map.remove(&ordinal);
    // Tracks the *registry*, not this device: another device may still have one
    // open, and a flag that said otherwise would route its stages to the pool.
    GUEST_ARENA_OPEN.store(!map.is_empty(), Ordering::Relaxed);
}

/// Open a generation on the device's guest arena.
///
/// One stage of a guest's pipeline per generation: the encode, a denoise step, a
/// decode tile — and, nested inside those, the transformer block or decoder
/// resnet that actually bounds the sum. Dropping the outermost fences the stream
/// and rewinds the cursor, so the next stage reuses the same bytes; dropping a
/// nested one releases to the cursor it opened at. That recycling is the whole
/// point, and why a guest needs an arena rather than a bump that only grows.
///
/// Every tensor allocated inside the generation dies with it. A value that has
/// to survive into the next stage must be copied out to the guest's ground
/// first; see `guest::arena::Stage` for the pairing.
pub fn begin_guest(stream: &Arc<CudaStream>) -> Result<Generation> {
    let ordinal = stream.context().ordinal();
    let map = guest_domains().lock().unwrap();
    let arena = map.get(&ordinal).ok_or_else(|| {
        candle::Error::Msg(
            "guest arena: no arena is open on this device — `open_guest_arena` runs at the start \
             of a drain and `close_guest_arena` at the end of it"
                .into(),
        )
    })?;
    let arena = arena.clone();
    drop(map);
    arena.generation(ordinal as u32, GUEST_ARENA)
}

/// Run `f` as one stage of a guest's pipeline, on the guest arena.
///
/// `x` is copied into the open generation so what `f` allocates inherits from
/// it, and `f`'s result is copied back out before the cursor rewinds. When no
/// arena is open — every caller that is not a guest — this is `f(x)` and nothing
/// else.
///
/// # Why a *block* is the stage, and a forward is not
///
/// [`begin_guest`] describes a stage as the encode, a denoise step, or a decode
/// tile, and for a small draw a step is one. It stops being one as the image
/// grows: a bump does not free inside a generation, so a generation holds the
/// **sum** of everything allocated in it, and a 1024×1024 denoise step is
/// thirty-four blocks and some six hundred allocations. Measured, that sum
/// saturated a 3,948 MiB arena and then a 7,020 MiB one — 3,937 and 6,993 —
/// and saturation is not a near miss: the first carve that fails yields an owned
/// tensor, nothing downstream of it can inherit, and the rest of the forward
/// reverts to the pool. One overflow undoes the whole arrangement.
///
/// A block is the stage that keeps the sum bounded. Its intermediates die with
/// it, the next block reuses the same bytes, and the arena holds one block
/// rather than thirty-four.
///
/// The cost is one copy in and one copy out per block — the hidden state, which
/// is the smallest tensor crossing that boundary and the only one that has to.
///
/// # Stages nest
///
/// "A block" is not one granularity for the whole pipeline. A transformer block
/// is the right stage for the denoise; a convolutional decoder's blocks are far
/// larger, and its own resnets are the unit that fits. Both are expressed the
/// same way, because [`Generation`] is LIFO and a nested one releases to the
/// cursor it opened at — so an inner stage bounds its sum inside an outer one
/// rather than being silently absorbed by it. Wrap whatever level actually
/// bounds the arena and let the nesting sort it out.
pub fn guest_stage<F>(x: &candle::Tensor, f: F) -> Result<candle::Tensor>
where
    F: FnOnce(&candle::Tensor) -> Result<candle::Tensor>,
{
    // The cheap negative, before any lock: this wraps model code shared with
    // callers that are not guests and never will be.
    if !GUEST_ARENA_OPEN.load(Ordering::Relaxed) {
        return f(x);
    }
    let Ok(cuda) = x.device().as_cuda_device() else {
        return f(x);
    };
    let stream = cuda.cuda_stream();
    let Ok(generation) = begin_guest(&stream) else {
        // No arena on *this* device, though one is open elsewhere.
        return f(x);
    };
    let staged = stage_in(x, cuda, &generation)?;
    let out = f(&staged)?;
    drop(staged);
    let inner = Arc::clone(&generation.inner);
    let (high, enclosed) = {
        let i = inner.lock().unwrap();
        (i.cursor, i.live > 1)
    };
    if !enclosed {
        // The outermost generation: its drop rewinds the whole arena and moves
        // the epoch, so nothing carved here can outlive it. The result is
        // copied out to the pool first.
        //
        // `empty` rather than `zeros`: `slice_set` below writes every element,
        // so a memset here would be a second full-width pass over the exact
        // bytes about to be stamped (principle 6).
        let survivor = candle::Tensor::empty(out.shape(), out.dtype(), out.device())?;
        survivor.slice_set(&out, 0, 0)?;
        drop(out);
        drop(generation);
        return Ok(survivor);
    }
    // **Carried, in the enclosing generation.** The stage releases to its mark
    // first — its result still sits in the released range, below `high`, and
    // nothing is carved there before the copy below reads it — and the result
    // is then carried at the top of the span, above `high`, where the
    // enclosing generation owns it. So a stage's result never leaves the
    // arena: it lives exactly as long as the stage around it, and the ops that
    // read it carve where it does.
    let ticket = WaveTicket {
        domain: generation.domain,
        arena: generation.arena,
        epoch: generation.ticket().epoch,
    };
    drop(generation);
    let survivor = carry_out(&out, cuda, &inner, ticket, high)?;
    drop(out);
    Ok(survivor)
}

/// Copy `out` into a carry at the top of `inner`'s span, above `high`.
pub(super) fn carry_out(
    out: &candle::Tensor,
    cuda: &candle::CudaDevice,
    inner: &Mutex<Inner>,
    ticket: WaveTicket,
    high: usize,
) -> Result<candle::Tensor> {
    let dst = carried(out.dims(), out.dtype(), cuda, inner, ticket, high)?;
    dst.slice_set(out, 0, 0)?;
    Ok(dst)
}

/// An uninitialised tensor carried at the top of `inner`'s span, above `high`,
/// owned by the generation `ticket` names.
pub(super) fn carried(
    dims: &[usize],
    dtype: candle::DType,
    cuda: &candle::CudaDevice,
    inner: &Mutex<Inner>,
    ticket: WaveTicket,
    high: usize,
) -> Result<candle::Tensor> {
    use candle::cuda_backend::wave_provenance::{exhausted, LeaseOrigin};

    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let ptr = {
        let mut i = inner.lock().unwrap();
        match carry_locked(&mut i, bytes, 256, high) {
            Some(ptr) => ptr,
            None => {
                let span = SpanState {
                    cursor: high + i.carry,
                    capacity: i.capacity,
                };
                return Err(exhausted(ticket, bytes, span));
            }
        }
    };
    let dev = candle::Device::Cuda(cuda.clone());
    // SAFETY: the range was carried for the generation `ticket` names, which
    // is still open — its caller holds it — and is released only when that
    // generation closes, after this tensor and everything read from it.
    unsafe {
        candle::Tensor::from_leased_cuda_ptr(
            ptr,
            dtype,
            dims.to_vec(),
            &dev,
            LeaseOrigin::Wave(ticket),
        )
    }
}

/// Run `f` over `items` in turn, each a stage reading the last one's result —
/// a model's block loop.
///
/// **Two carries for the whole chain, not one per stage.** [`guest_stage`]
/// carries each result until the generation around it closes, which is right
/// for a stage whose result something later reads, and wrong for a block loop:
/// block `n`'s result is dead once block `n + 1` has read it, yet forty of
/// them stood until the step closed — a gigabyte at 1024×768, past an arena
/// the claim priced for one block. Here each stage writes into the carry the
/// one before it read from, so the loop holds two hidden states whatever its
/// length. Every stage must keep its input's shape, as a residual block does.
///
/// With no enclosing generation — nothing to own a carry — each item is an
/// ordinary [`guest_stage`].
pub fn guest_chain<'a, T: 'a, I, F>(
    x: &candle::Tensor,
    items: I,
    mut f: F,
) -> Result<candle::Tensor>
where
    I: IntoIterator<Item = &'a T>,
    F: FnMut(&candle::Tensor, &'a T) -> Result<candle::Tensor>,
{
    let plain = |x: &candle::Tensor, items: I, f: &mut F| -> Result<candle::Tensor> {
        let mut h = x.clone();
        for item in items {
            h = guest_stage(&h, |h| f(h, item))?;
        }
        Ok(h)
    };
    if !GUEST_ARENA_OPEN.load(Ordering::Relaxed) {
        return plain(x, items, &mut f);
    }
    let Ok(cuda) = x.device().as_cuda_device() else {
        return plain(x, items, &mut f);
    };
    let ordinal = cuda.cuda_stream().context().ordinal();
    let Some(arena) = guest_domains().lock().unwrap().get(&ordinal).cloned() else {
        return plain(x, items, &mut f);
    };
    let (live, epoch, cursor) = {
        let i = arena.inner.lock().unwrap();
        (i.live, i.epoch, i.cursor)
    };
    if live == 0 {
        return plain(x, items, &mut f);
    }
    // Owned by the generation open now, as a stage's carried result is.
    let ticket = WaveTicket {
        domain: ordinal as u32,
        arena: GUEST_ARENA,
        epoch,
    };
    let mut cur = carried(x.dims(), x.dtype(), cuda, &arena.inner, ticket, cursor)?;
    let mut next = carried(x.dims(), x.dtype(), cuda, &arena.inner, ticket, cursor)?;
    cur.slice_set(x, 0, 0)?;
    for item in items {
        let stage = arena.generation(ordinal as u32, GUEST_ARENA)?;
        // `cur` carries the open generation's epoch, so what the stage derives
        // from it carves at the cursor — inside this stage — and goes when it
        // closes.
        let out = f(&cur, item)?;
        if out.dims() != cur.dims() || out.dtype() != cur.dtype() {
            candle::bail!(
                "guest chain: a stage turned {:?} {:?} into {:?} {:?} — a chain's stages keep \
                 their input's shape",
                cur.dims(),
                cur.dtype(),
                out.dims(),
                out.dtype()
            );
        }
        // Into the carry the stage did not read: the one it read is still
        // under its kernels until the stream reaches this copy.
        next.slice_set(&out, 0, 0)?;
        drop(out);
        drop(stage);
        std::mem::swap(&mut cur, &mut next);
    }
    Ok(cur)
}

/// Copy `x` into the open guest generation, so ops reading it carve there.
///
/// Inheritance runs along the operand chain — a matmul takes its ticket from the
/// lhs — so a stage whose input is owned allocates everything from the pool
/// however well the arena is sized. This is where each stage's chain starts.
///
/// **The ticket carries `generation`'s real epoch, not [`GUEST_ANY_EPOCH`].**
/// The any-epoch licence exists for the *weights*, which are placed once and
/// read by every stage, and it is the one thing that must not be handed to a
/// tensor that names a range: a stage's chain has to stop resolving when the
/// stage's arena rewinds, or a ticket outliving its scope would carve from
/// ground the arena has already given back.
pub(super) fn stage_in(
    x: &candle::Tensor,
    cuda: &candle::CudaDevice,
    generation: &Generation,
) -> Result<candle::Tensor> {
    use candle::cuda_backend::wave_provenance::{exhausted, wave_alloc, LeaseOrigin, WaveCarve};

    let ticket = generation.ticket();
    let bytes = x.elem_count() * x.dtype().size_in_bytes();
    let ptr = match wave_alloc(ticket, bytes, 256) {
        WaveCarve::Carved(ptr) => ptr,
        WaveCarve::Closed => return Ok(x.clone()),
        WaveCarve::Exhausted(span) => return Err(exhausted(ticket, bytes, span)),
    };
    let dev = candle::Device::Cuda(cuda.clone());
    // SAFETY: the range was carved from the generation open above, which the
    // caller drops only after this tensor and everything derived from it.
    let dst = unsafe {
        candle::Tensor::from_leased_cuda_ptr(
            ptr,
            x.dtype(),
            x.dims().to_vec(),
            &dev,
            LeaseOrigin::Wave(ticket),
        )
    }?;
    dst.slice_set(x, 0, 0)?;
    Ok(dst)
}

/// `(cursor, peak, capacity)` for the guest arena on `ordinal`, or `None` when
/// no drain is running. The peak is what a guest's activation claim should be
/// sized from, measured rather than guessed.
pub fn guest_domain_stats(ordinal: usize) -> Option<(usize, usize, usize)> {
    guest_domains()
        .lock()
        .unwrap()
        .get(&ordinal)
        .map(|a| a.stats())
}
