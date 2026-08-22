//! Descriptor-table staging for the hot-path kernels.
//!
//! Hot-path invariant 2b has the kernels take their per-row operands as a table
//! of `{ptr, stride, len}` rather than a packed block, which is what removed the
//! copies. But the table itself has to reach the device, and `ncu` says that —
//! not the kernels — is where the time went:
//!
//! | shape                          | kernel  | wall    |
//! |--------------------------------|---------|---------|
//! | `compressor_pool` decode ×1    | 5.2 us  | 15.6 us |
//! | `rows_scatter` 64 sessions     | 2.9 us  | 26.8 us |
//!
//! A `Tensor::from_vec` of a 928-word table measured **7.6 us** against a 5.2 us
//! kernel: it allocates a fresh device buffer per call AND copies from pageable
//! host memory, and under WDDM each of those driver calls crosses the
//! kernel-mode scheduler.
//!
//! # The wave arena
//!
//! Neither cost is necessary, because the wave already runs inside a
//! [`Generation`] of the shared [`PinnedStager`] — a bump arena mapped
//! `DEVICEMAP | WRITECOMBINED`, which the paged-attention path already uses for
//! its slice and header metadata. Allocating from it is a pointer increment, and
//! `submit` on a bump buffer hands back a device-visible address into the SAME
//! pinned pages: the GPU reads the table live over PCIe, so there is no copy, no
//! allocation and no driver call at all. Write-combined is exactly right for the
//! access pattern — the host only writes the table, the device only reads it.
//!
//! Freeing is by lifetime: the arena resets when the last `Generation` drops, so
//! a table stays valid for precisely as long as the wave that built it. That is
//! also why a generation may NOT be opened per call — dropping the last one
//! synchronises the stream, which is the same stall that made a naive pinned
//! staging buffer 5× worse (below).
//!
//! There is NO second staging path. A caller outside a wave — the streamed
//! reference emit, a turn-seal `close`, the gates — does not LACK an arena, it
//! simply has not opened a SCOPE yet, so it opens one with [`scope`]. Generations
//! are refcounted: the arena resets only when the LAST one drops, so a scope
//! opened inside a wave nests harmlessly (no reset, no sync) and a standalone one
//! resets at its own end, when nothing else is in flight.
//!
//! That distinction matters because the alternative was an `Option<&Generation>`
//! with a copy-based fallback, and a shared fallback buffer is only sound when at
//! most ONE table is staged per launch — which stopped being true the moment
//! `gather_corpus_batched` staged its pointer table and its metadata before a
//! single kernel. The second overwrote the first and the kernel read the wrong
//! address (`CUDA_ERROR_ILLEGAL_ADDRESS`, half the gallery gates). Staging
//! lifetime cannot be inferred from call order; the arena gives every `alloc` a
//! distinct region, so the hazard cannot arise.
//!
//! # Measured dead end — do not re-attempt
//!
//! **A dedicated pinned staging buffer made the pool 15 → 80 us, a 5×
//! regression.** `PinnedHostSlice` guards reuse with an event and
//! `as_mut_slice` synchronises on it, which with one reusable buffer drains the
//! entire queued stream every call and throttles the host to GPU rate. A ring
//! deep enough to dodge that cannot be sized, because the host legitimately runs
//! hundreds of iterations ahead of the device. The arena avoids the problem
//! rather than fighting it: nothing is reused until the generation ends.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle::cuda_backend::DeviceId;
use candle::quantized::pinned_staging::{Generation, GpuBuf, PinnedStager};
use candle::{Device, Result};

/// Bytes of arena one wave draws for descriptor tables — the addition this
/// module makes to a budget that previously covered only the paged metadata.
///
/// Per compression layer a wave stages the two compressor pools (attention and
/// indexer), the gallery append's scatter, the corpus gather, the BDP recall,
/// the two-stage selector, the decode path's compressed-index expansion, the
/// arena writeback, and the batched prefill's slot metadata.
///
/// * A pool table is `POOL_GROUP_WORDS·G + POOL_SEG_WORDS·(G·segs)` words. At the
///   decode streamer's shape a group is `2·ratio` one-row segments, so `segs` is
///   `2·ratio`; that is the widest form, since a bulk assemble describes the same
///   groups with one or two wide segments instead.
/// * A scatter table is `ROWS_SCATTER_WORDS · runs`, with up to six runs per slot
///   (positions, signs, keys and the three latent regions).
/// * `max_tokens` is the wave's widest per-slot token count — one prompt's
///   length during prefill, one for a decode step. The writeback and prefill
///   metadata are the only per-TOKEN draws; everything else scales with slots.
///
/// Bump-allocated, so the cost is address space rather than work, and it is freed
/// when the wave's `Generation` drops. Computed rather than assumed so the draw is
/// KNOWN — checked against the arena at the wave's `begin_stager_generation`
/// rather than discovered when an overflow arena appears.
/// The stager's default arena, mirroring `DEFAULT_ARENA_SIZE` in
/// `candle-core/src/quantized/pinned_staging.rs`. The budget assert compares
/// against it; `CANDLE_ARENA_MB` can only raise it, so the check stays
/// conservative.
pub const STAGER_ARENA_BYTES: usize = 128 * 1024 * 1024;

/// Descriptor word size. Pointer tables are `i64`; the count/offset tables are
/// `u32` and are converted with `div_ceil` so a table that does not fill a whole
/// word still costs one.
const WORD: usize = std::mem::size_of::<i64>();

pub fn wave_desc_bytes(slots: usize, layers: usize, ratio: usize, max_tokens: usize) -> usize {
    use candle_kernels::paged_latent::GLUE_SCATTER_WORDS;
    use candle_kernels::simple::comp_idx::COMP_IDX_SLOT_WORDS;
    use candle_kernels::simple::compressor_pool::{POOL_GROUP_WORDS, POOL_SEG_WORDS};
    use candle_kernels::simple::rows_scatter::ROWS_SCATTER_WORDS;
    let w32 = |words: usize| (words * std::mem::size_of::<u32>()).div_ceil(WORD);
    let pool = POOL_GROUP_WORDS * slots + POOL_SEG_WORDS * slots * 2 * ratio;
    let scatter = ROWS_SCATTER_WORDS * slots * 6;
    // The corpus gather's `[nope|scale|rope|pos|gids]` pointer table plus its
    // `[out_off|cnt]` metadata, and the BDP recall's per-gallery sign table.
    let gather = 5 * slots + 2 * slots;
    let bdp = slots;
    // The two-stage selector's `off`/`cnt` segment tables — `u32` a slot each,
    // staged once and shared by the recall and the top-M.
    let select = w32(2 * slots);
    // The decode path's compressed-index expansion: `{offset, count}` a slot.
    let comp_idx = w32(COMP_IDX_SLOT_WORDS * slots);
    // The arena writeback: one run per slot, each staging a `{slice, in_blk}`
    // u32 pair PER TOKEN OF THAT RUN — so the index tables scale with
    // slots × tokens, not with either alone. This is the draw that dominates a
    // long prefill and is why the budget takes `max_tokens`.
    let glue = GLUE_SCATTER_WORDS * slots + slots * w32(2 * max_tokens);
    // The batched prefill's per-query slot selector (one u32 per token across
    // the WHOLE prompt fleet) and its per-seq diagonal metadata (four a slot).
    let prefill_meta = w32(slots * max_tokens) + w32(4 * slots);
    // Per layer: two pools (attention + indexer), one append scatter, one corpus
    // gather, one recall, one selector, one index expansion, one writeback and
    // the prefill metadata.
    (2 * pool + scatter + gather + bdp + select + comp_idx + glue + prefill_meta) * layers * WORD
}

/// A staged descriptor table. Holds the arena buffer that backs the address, so
/// the caller must keep it in scope until the launch that reads it is enqueued.
pub struct StagedDesc {
    ptr: u64,
    _arena: GpuBuf,
}

impl StagedDesc {
    /// Device address of the table.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
}

/// The descriptor arena for one device, created on first use.
///
/// Separate from the KV backing's stager: this one's lifetime is owned by the
/// scopes below rather than by the paged-metadata path, so a caller that has no
/// wave still has somewhere to stage from.
/// Non-CUDA devices get the stager's device-less mode, so a CPU reference path
/// can open a scope like anything else — it simply never stages, because the
/// pool's CPU arm computes without a descriptor.
fn stager(device: &Device) -> Result<PinnedStager> {
    static STAGERS: OnceLock<Mutex<HashMap<Option<DeviceId>, PinnedStager>>> = OnceLock::new();
    let key = match device {
        Device::Cuda(d) => Some(d.id()),
        _ => None,
    };
    let map = STAGERS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map
        .lock()
        .map_err(|_| candle::Error::Msg("descriptor stager map poisoned".into()))?;
    Ok(map
        .entry(key)
        .or_insert_with(|| match device {
            Device::Cuda(d) => PinnedStager::new(d),
            other => PinnedStager::new_from_device(other),
        })
        .clone())
}

/// Open a staging scope. Tables staged while it lives stay valid; the arena
/// resets when the LAST live scope drops.
///
/// The wave opens one for its whole layer loop. Anything else — a turn-seal
/// `close`, the streamed reference, a gate — opens its own around the operation
/// it is doing. Nested scopes are free: only the outermost drop resets, so a
/// scope taken inside a wave neither resets the arena nor synchronises.
pub fn scope(device: &Device) -> Result<Generation> {
    Ok(stager(device)?.begin_generation())
}

/// Stage `words` where the device can read it: bump-allocated from `generation`'s
/// arena and read IN PLACE over PCIe — no allocation, no copy, no driver call.
pub fn stage(words: &[i64], generation: &Generation) -> Result<StagedDesc> {
    stage_slice(words, generation)
}

/// [`stage`] for a table of any plain-old-data element — `u32` counts and
/// offsets as readily as `i64` pointers. The arena is untyped bytes and the
/// kernels reinterpret what they are given, so the element type only decides how
/// many bytes are written.
pub fn stage_slice<T: Copy>(words: &[T], generation: &Generation) -> Result<StagedDesc> {
    if words.is_empty() {
        candle::bail!("descriptor stage: empty table");
    }
    let bytes = std::mem::size_of_val(words);
    let mut buf = generation.alloc(bytes)?;
    // The arena is untyped bytes; the table is written straight into the pinned
    // pages the GPU will read.
    //
    // SAFETY: `T: Copy` with no padding-sensitive reads — the destination is a
    // byte buffer the device reinterprets with the same layout the kernel was
    // compiled against, and `size_of_val` bounds the read exactly.
    buf.as_mut_slice()[..bytes].copy_from_slice(unsafe {
        std::slice::from_raw_parts(words.as_ptr() as *const u8, bytes)
    });
    let gpu = generation.submit(buf)?;
    Ok(StagedDesc {
        ptr: gpu.dev_ptr(),
        _arena: gpu,
    })
}

