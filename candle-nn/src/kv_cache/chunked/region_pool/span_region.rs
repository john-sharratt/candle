//! A region of the reservation held by a tenant other than a KV arena.
//!
//! Long-lived device buffers of region size, such as recurrent state or a
//! gallery slab, are claimed from the reservation. Allocations outside the
//! reservation compete with it for the card, and on WDDM the loser is demoted
//! to host memory. A claim moves the live end that the transient tier stands
//! against, so it takes the same arena window an arena does and is refused
//! during a forward rather than carved from ground the tier occupies. The span
//! tenant count is maintained by the handle's construction and drop, so it
//! cannot drift from the handles that exist.

use super::*;

/// One region of the reservation, held by a tenant that is not a KV arena.
///
/// The reservation is the budget. Anything this process allocates outside it —
/// through `cudaMalloc`, through the async pool, through a bare
/// `alloc_zeros` — is competing with it for the same card, and on WDDM the
/// loser is not an error but a demotion to host RAM: measured at 3.7 GiB on the
/// 3.6-35B, which cost 17x on decode with nothing anywhere reporting it. Memory
/// inside the span cannot lose that fight, because its granules are pinned
/// device allocations the driver may not migrate.
///
/// So a long-lived device buffer of region size belongs here rather than in the
/// pool. Dropping the handle returns the region, exactly as an arena's does.
pub struct SpanRegion {
    pub(super) inner: RegionHandle,
}

impl SpanRegion {
    /// Wrap a claimed region as a span tenant's, counting it as one.
    ///
    /// The only constructor, so `span_live` cannot disagree with the handles that
    /// exist — which is the property the efficiency figure needs, because it is
    /// otherwise formed by subtracting two populations sampled behind two locks.
    pub(super) fn adopt(inner: RegionHandle) -> Self {
        let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pool) = map.get_mut(&inner.ordinal) {
            pool.span_live += 1;
        }
        drop(map);
        Self { inner }
    }

    /// Device address of the region's first byte.
    pub fn base(&self) -> u64 {
        self.inner.base()
    }

    /// The region's position in the span, ascending with address.
    pub fn index(&self) -> usize {
        self.inner.index()
    }

    /// Bytes in a region — the unit this allocator deals in.
    pub const fn bytes() -> usize {
        REGION_BYTES
    }
}

impl Drop for SpanRegion {
    fn drop(&mut self) {
        // Before the inner handle's own drop returns the region to the free list,
        // and on the same lock it takes — so a reader between the two sees the
        // region as neither a span tenant's nor free, never as both.
        let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pool) = map.get_mut(&self.inner.ordinal) {
            pool.span_live = pool.span_live.saturating_sub(1);
        }
    }
}

impl std::fmt::Debug for SpanRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanRegion")
            .field("base", &self.inner.base())
            .field("index", &self.inner.index())
            .finish()
    }
}

/// Claim one region of the reservation for a non-KV tenant, or `None` when the
/// KV side has none spare.
///
/// `None` is the same pressure signal [`claim_region`] gives an arena, and the
/// caller's answer is the same: shed something, or do without. It is not an
/// error — a tenant that cannot get a region is expected to say so and carry on
/// with fewer, which is what the gallery's own cap already does.
///
/// The region's bytes are zero on return.
///
/// # It takes the arena window, and must
///
/// A claim moves `live_end`, and [`try_place`] puts the transient tier flush
/// against `live_end` with **no gap above it** — a placement that is only sound
/// because every claim used to be an arena creation, which
/// [`enter_arena_window`] holds off until the gap between forwards. Claiming
/// straight from `claim_region` skips that gate, so a region taken during a
/// forward is carved out of ground the standing tier already occupies: the
/// tenant's buffer and the wave's scratch become the same bytes, and whichever
/// kernel reads them next dies somewhere else entirely.
///
/// That is not hypothetical. Gallery slabs and recurrent state claimed here
/// without the window, and the hot→warm migrate path — an innocent reader of
/// KV arena memory — took `CUDA_ERROR_ILLEGAL_ADDRESS` and
/// `CUDA_ERROR_LAUNCH_FAILED` in the hundreds per run. Under
/// `CUDA_LAUNCH_BLOCKING=1` it vanished, which is what a corrupted-address race
/// looks like from the outside and is why it read as a fault in the migrate
/// kernels for so long.
///
/// So a span tenant is bound by the same rule as an arena: **allocate between
/// forwards.** Inside one, this refuses rather than corrupts.
pub fn claim_span_region(
    device: &candle::Device,
    tenant: &'static str,
) -> Result<Option<SpanRegion>> {
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("claim_span_region: the reservation is a CUDA allocation");
    };
    // Held across the claim, so a forward cannot open between the gate and the
    // frontier moving.
    let _window = super::super::bump_arena::enter_arena_window(cuda.cuda_stream(), tenant)?;
    Ok(claim_region(&cuda.cuda_stream())?.map(SpanRegion::adopt))
}

/// An open arena window, for a tenant claiming SEVERAL regions at once.
///
/// [`claim_span_region`] takes the window per call, which is right for a tenant
/// that wants one region and wrong for one that wants eight: entering the window
/// hands back a standing tier, so a store built a region at a time releases and
/// re-places the tier once per region, and every recycled claim inside that
/// carries a device-wide quiesce. The churn is not a correctness problem — the
/// gate holds either way — but it is enough of one to reach the WDDM watchdog,
/// which terminates the launch and poisons the context just as surely.
///
/// Held open for the whole construction instead, a tenant pays for one window
/// and one tier handback however many regions it takes.
pub struct SpanClaims {
    pub(super) device: candle::CudaDevice,
    /// The window itself. Dropped with this, which is what re-arms the tier for
    /// the next forward.
    pub(super) _window: super::super::bump_arena::ArenaWindow,
}

impl SpanClaims {
    /// Open the window. Refuses inside a forward, for the reason on
    /// [`claim_span_region`].
    pub fn open(device: &candle::Device, tenant: &'static str) -> Result<Self> {
        let candle::Device::Cuda(cuda) = device else {
            candle::bail!("SpanClaims: the reservation is a CUDA allocation");
        };
        let _window = super::super::bump_arena::enter_arena_window(cuda.cuda_stream(), tenant)?;
        Ok(Self {
            device: cuda.clone(),
            _window,
        })
    }

    /// One more region, or `None` when the KV side has none spare.
    pub fn claim(&self) -> Result<Option<SpanRegion>> {
        Ok(claim_region(&self.device.cuda_stream())?.map(SpanRegion::adopt))
    }

    /// `k` consecutive regions — one block of `k × REGION_BYTES` — or `None`
    /// when no such run is free and the weight side will not concede one.
    pub fn claim_run(&self, k: usize) -> Result<Option<Vec<SpanRegion>>> {
        Ok(claim_region_run(&self.device.cuda_stream(), k)?
            .map(|run| run.into_iter().map(SpanRegion::adopt).collect()))
    }

    /// `k` consecutive regions ending at or below region `below`, from ground the
    /// KV side already has spare — never bought — or `None` when no such run is
    /// free. How a tenant standing high moves down onto ground packing freed.
    pub fn claim_run_below(&self, k: usize, below: usize) -> Result<Option<Vec<SpanRegion>>> {
        let stream = self.device.cuda_stream();
        Ok(
            with_pool(&stream, |pool| try_claim_run(pool, &stream, k, below))?
                .map(|run| run.into_iter().map(SpanRegion::adopt).collect()),
        )
    }
}

/// [`claim_region`] for `k` consecutive regions: the lowest run that fits, bought
/// from the weight side when the KV side has none.
pub(crate) fn claim_region_run(
    stream: &std::sync::Arc<CudaStream>,
    k: usize,
) -> Result<Option<Vec<RegionHandle>>> {
    if let Some(run) = with_pool(stream, |pool| try_claim_run(pool, stream, k, usize::MAX))? {
        return Ok(Some(run));
    }
    let tier_standing = with_pool(stream, |pool| Ok(pool.transient_base.is_some()))?;
    if tier_standing || buy_ground(stream, k.max(KV_BUY_STEP))? == 0 {
        return Ok(None);
    }
    with_pool(stream, |pool| try_claim_run(pool, stream, k, usize::MAX))
}

/// Whether span regions may be claimed on `stream`'s device yet: not while a
/// model is loading into the dense block, whose right edge every region is
/// measured from (`claim_dense` refuses once one exists).
pub(crate) fn regions_open(stream: &std::sync::Arc<CudaStream>) -> bool {
    with_pool(stream, |pool| Ok(pool.dense_frozen || !pool.dense_capable)).unwrap_or(false)
}

/// Why a [`claim_span_region`] came back empty, for an error message that can
/// name the difference.
///
/// The two refusals have opposite fixes and a caller that cannot tell them
/// apart reports the wrong one: a standing tier means the ground exists but
/// belongs to a running wave (allocate earlier, or run a narrower wave), while
/// exhaustion means the weight side would not concede (there is genuinely no
/// room). Guessing between them cost an hour.
pub fn span_region_refusal(device: &candle::Device) -> &'static str {
    let candle::Device::Cuda(cuda) = device else {
        return "not a CUDA device";
    };
    let stream = cuda.cuda_stream();
    let standing = with_pool(&stream, |pool| Ok(pool.transient_base.is_some())).unwrap_or(false);
    if standing {
        "a wave transient tier is standing — this ran INSIDE a forward, where the \
         ground above the tier belongs to the wave and no weight-side concession \
         reaches it. Allocate before the wave opens."
    } else {
        "no tier stands, so the span is genuinely full — the weight side is at its \
         own floor and could not concede ground."
    }
}
