//! Claims each device's reservation on first use and gives access to its pool.
//!
//! Every KV backing claims its device's reservation through here, so the first
//! call creates it and later calls find it. A process that loads a second model
//! finds the span already claimed, and that model's weights go to the ordinary
//! pool, which is why `ensure_reservation` reports whether it created the
//! reservation. The reservation is created early only on a measured card,
//! because it is created once and never resized. A span too small for the fixed
//! blocks and one wave's tier is refused outright.

use super::*;

/// The smallest span that can run a forward whose fixed block is `persist`
/// bytes: that block plus one wave's transient tier, with nothing left for KV.
///
/// Two callers, and they must agree. `create_with_headroom` will not concede
/// load headroom past this point, and the same function refuses a span that
/// comes up short of it — so a disagreement would let the first hand back ground
/// the second then rejects, turning an optimisation into a startup failure.
pub(super) fn span_minimum_bytes(persist: usize) -> usize {
    persist + MAX_WAVE_TRANSIENT_BYTES
}

pub(super) fn pools() -> &'static Mutex<HashMap<usize, RegionPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, RegionPool>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn with_pool<R>(
    stream: &std::sync::Arc<CudaStream>,
    f: impl FnOnce(&mut RegionPool) -> Result<R>,
) -> Result<R> {
    let mut map: MutexGuard<'_, HashMap<usize, RegionPool>> =
        pools().lock().unwrap_or_else(|e| e.into_inner());
    let ordinal = stream.context().ordinal();
    let pool = match map.entry(ordinal) {
        Entry::Occupied(o) => o.into_mut(),
        // No headroom on the lazy path: reaching here means nothing called
        // `ensure_reservation`, so no model is loading and there is no repack
        // peak to leave room for. That is every test, and the CPU-device paths.
        Entry::Vacant(v) => v.insert(RegionPool::create_with_headroom(stream, 0)?),
    };
    f(pool)
}

/// Claim this device's reservation if it does not exist yet.
///
/// Called once per KV backing so the claim is a startup cost and the free-region
/// count — which admission reads before any arena exists — always has an answer.
pub(crate) fn ensure(stream: &std::sync::Arc<CudaStream>) -> Result<()> {
    with_pool(stream, |_| Ok(()))
}

/// Claim the reservation **before the model loads**, its top `load_headroom`
/// bytes opened as the load's own scratch until [`freeze_dense`].
///
/// This is the call that makes the weights span tenants at all. The span used to
/// be sized from `usable()` *after* the weights were resident, so it was
/// measured against the hole they had already taken from the CUDA pool and could
/// not, even in principle, contain them. Claiming first inverts that: the span
/// is sized from the whole card, and the weights are carved out of it by
/// [`claim_dense`] as they arrive.
///
/// Nothing has to predict the model's size. The dense block grows to whatever
/// the checkpoint needs and the region count is derived from the remainder at
/// [`freeze_dense`]. The one quantity the caller does supply is `load_headroom`
/// — see [`RegionPool::create_with_headroom`] — and that is an arithmetic bound
/// read off the GGUF header, not a guess.
///
/// # It must be first, and it reports whether it was
///
/// `true` when this call created the reservation — the load is about to get the
/// ordering this function exists for. `false` when one already existed, which is
/// **not an error**: a process that loads a second model finds the span already
/// claimed and sized, and its weights go to the pool exactly as they did before
/// any of this. Only the first model through can be a span tenant, because only
/// its load happens while the span is still unclaimed.
///
/// Returned rather than bailed so the caller can say which of the two happened.
/// Failing instead would turn "this is the second model in the process" — every
/// multi-model test binary — into a load failure, and swallowing it silently
/// would leave a reader unable to explain why `dense_bytes` was zero.
pub fn ensure_reservation(device: &candle::Device, load_headroom: usize) -> Result<bool> {
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("ensure_reservation: the reservation is a CUDA allocation");
    };
    let stream = cuda.cuda_stream();
    let ordinal = stream.context().ordinal();

    // **Claiming early is only right once the card has been measured.** Without
    // a VRAM governor `span_target` answers with the fixed `TEST_SPAN_BYTES`
    // constant, and claiming here would freeze that constant in as the span for
    // the process — there is no second chance, because the reservation is
    // created once.
    //
    // Loaders differ on when the governor appears: some install it before they
    // read a tensor, others part-way through their own load. Qwen3-30B-A3B is
    // the second kind, and calling this at the top of its loader took a 73 GiB
    // card down to a **185-region, 2.9 GiB span** — enough for the model to load
    // and then fail with "Expert cache full, cannot evict (all pinned)", a
    // message about eviction several steps from the span that was never big
    // enough. Declining here leaves the reservation to be created lazily and
    // correctly later, exactly as it was before.
    if candle::vram::get(ordinal).is_none() {
        return Ok(false);
    }

    let mut map: MutexGuard<'_, HashMap<usize, RegionPool>> =
        pools().lock().unwrap_or_else(|e| e.into_inner());
    match map.entry(ordinal) {
        Entry::Occupied(_) => Ok(false),
        Entry::Vacant(v) => {
            v.insert(RegionPool::create_with_headroom(&stream, load_headroom)?);
            Ok(true)
        }
    }
}

/// Claim the reservation if it does not exist, and report the span's right edge.
///
/// The address an expert cache builds its [`super::weight_zone::WeightZone`]
/// from: slots fill leftward from here.
pub fn span_end(stream: &std::sync::Arc<CudaStream>) -> Result<u64> {
    with_pool(stream, |pool| Ok(pool.span_end()))
}
