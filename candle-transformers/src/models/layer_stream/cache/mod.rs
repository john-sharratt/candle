//! The layer cache: residency, the tiers that hold each layer's bytes, and the
//! transfers between them.
//!
//! Each forward calls `ensure` for a layer and then `prefetch` for the layers that
//! follow it. `ensure` moves the wave, joins that layer's fence if a copy is in
//! flight, and hands back its matmuls. The wave moves inside `ensure` because the
//! two must always be called together, in that order. A fence per layer lets the
//! wave wait only for the layer it needs rather than for a whole plan's copies. The
//! cache is generic over how a slot is presented, so it can be tested with no
//! model at all.
//!
//! - `cache_ops`: filling the tiers, moving the wave and issuing copies.
//!
//! - `tests_streaming`: byte identity across zones and cold reads.

use candle::cuda_backend::cudarc::driver::{CudaEvent, CudaStream};
use candle::quantized::Int8Mode;
use candle::{CudaDevice, Result};
use std::sync::Arc;

use super::descriptor::LayerImage;
use super::pack::{LayerPack, PackRead};
use super::residency::{LayerResidency, LoadOp, PlanScratch, Residence};
use super::view::{build_layer_view, StreamedLayer};
use super::warm::{warm_membership, warm_slots_for};
use crate::models::expert_lre::pinned::WarmPool;
use crate::models::expert_lre::warm_tier::addressable_pinned_slots;
use crate::models::layer_stream::zone::{LayerPlacement, ZonePlan};
use candle::vram::{gpu_addressable_room, PinnedUse};

pub use super::residency::COMMITTED_DEPTH;

/// A failed [`LayerCache::issue`], and whether the slot survived it.
///
/// The distinction is the whole reason this type exists: the rollback may put
/// the evicted tenant back only when nothing was enqueued, and must clear the
/// slot's view when something was.
enum Failed {
    /// The H2D was never enqueued — the slot still holds its previous tenant,
    /// bytes and view both.
    Untouched(candle::Error),
    /// The copy is in flight over the slot. The previous tenant is gone.
    Clobbered(candle::Error),
}

/// Buffers in the cold-read staging ring.
///
/// Each is one record wide, so the ring costs `STAGING_SLOTS × stride` of pinned
/// host memory — ~1.1 GiB at the 27B's ~290 MB record. That is real, and it is
/// why this is four rather than the plan's full width: the ring only has to
/// cover the reads *in flight at once*, and a read is finished with its buffer
/// the moment the H2D behind it lands. Four gives the copy stream three landed
/// buffers of slack before a fill has to wait on one.
pub const STAGING_SLOTS: usize = 4;

/// Counters for the gate's report.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LayerCacheStats {
    /// The layer was already resident when the wave reached it — no fence, no
    /// bytes moved.
    pub hits: usize,
    /// Promoted from the pinned host tier.
    pub warm_hits: usize,
    /// Read from the pack.
    pub cold_reads: usize,
    /// The wave joined a layer's fence before reading it.
    ///
    /// **Not a measure of time lost.** A join whose copy has already landed
    /// costs nothing, and on a ring smaller than the model *every* streamed
    /// layer is joined by construction — the count is bounded below by the
    /// number of layers that had to stream, which §9.2 says is
    /// `total − resident` and no policy can reduce. What it does distinguish is
    /// the two regimes: a model that fits reports joins near zero, and a model
    /// that does not reports one per streamed layer.
    ///
    /// Measuring the *wait* would need the fence's elapsed time, which is a
    /// device round trip on the hot path and is deliberately not taken here.
    pub fence_joins: usize,
    pub evictions: usize,
    /// Opportunistic loads dropped because their transfer failed.
    pub abandoned: usize,
    /// Layers with a permanent address **right now**, not at the carve.
    ///
    /// State rather than a counter, and filled by [`LayerCache::stats`] at read
    /// time for that reason. It is here because the carve is the only residency
    /// anything printed, and the carve is not the number that decides
    /// throughput: the boundary grows into spare KV ground over the first
    /// forwards, so a zone that opens at 47 of 64 may be running at 64. Reading
    /// that off the transfer counters is an inference; this is the observation.
    pub homed: usize,
    /// Layers that still cross PCIe every forward. `0` means the zone holds the
    /// trunk and the streaming machinery is inert.
    pub streaming: usize,
}

/// Turns a slot's views into what the forward reads.
///
/// A trait rather than a bare `Fn` bound because the model's assembler is a
/// named type — it carries the per-layer residues and images — and a type that
/// has to be spelled in a struct field cannot be a closure. The blanket impl
/// keeps closures working, which is what lets the cache's own tests exercise it
/// with no model at all.
pub trait SlotAssembler<T> {
    fn assemble(&self, view: StreamedLayer, layer: usize) -> Result<T>;
}

impl<T, F: Fn(StreamedLayer, usize) -> Result<T>> SlotAssembler<T> for F {
    fn assemble(&self, view: StreamedLayer, layer: usize) -> Result<T> {
        self(view, layer)
    }
}

/// Where a layer's bytes come from when it is not resident.
enum Source {
    /// A pinned host slot — one H2D of contiguous bytes.
    Warm(usize),
    /// The pack, staged through a pinned buffer.
    Cold,
}

/// Residency plus the tiers and the stream that serve it.
///
/// Generic over what a slot's bytes are *presented* as. The cache moves bytes
/// and tracks residency; turning a slot into something a forward can read is
/// model knowledge, so it is supplied as `assemble` rather than known here.
/// The qwen35 lineage passes a closure that builds a whole `QuantLayer` (see
/// [`super::assemble`]), which is what lets the forward keep taking
/// `&QuantLayer` and never learn a slot exists. A test can pass the identity
/// and get the raw views back.
pub struct LayerCache<T, A>
where
    A: SlotAssembler<T>,
{
    device: CudaDevice,
    /// The copy stream. Separate from the compute stream so a transfer overlaps
    /// the layers already resident rather than serialising behind them.
    copy: Arc<CudaStream>,
    images: Vec<LayerImage>,
    /// The numeric mode the model was loaded for, so a rebuilt view reports the
    /// same path the loader chose. A slot holds a KO twin or a source quant, and
    /// which one it is decides whether the weight is fed q8a128 activations.
    mode: Int8Mode,
    residency: LayerResidency,
    pack: LayerPack,
    warm: WarmPool,
    /// `layer → warm slot`, for the static membership.
    warm_slot: Vec<Option<usize>>,
    /// Staging for a cold read whose layer has no warm slot.
    ///
    /// A **ring**, and it has to be. The read fills a buffer on the host and the
    /// upload out of it is asynchronous on the copy stream, so the buffer is
    /// still being read by the DMA after `issue` returns. One buffer would be
    /// overwritten by the next cold read in the same plan while the previous
    /// transfer was mid-flight, and the layer that transfer was carrying would
    /// land holding a different layer's bytes.
    ///
    /// That is not hypothetical: it is what a single buffer did. The bug hid for
    /// as long as the warm tier happened to cover every streamed layer (`cold 0`,
    /// so the path never ran twice in a plan); sizing the warm tier honestly
    /// against host RAM introduced 132 cold reads per config on the 27B and the
    /// model started producing garbage logits within three configs.
    staging: WarmPool,
    /// `staging slot → the event marking its last upload complete`. A slot is
    /// reused only after this is joined, which is what makes the ring a ring
    /// rather than a rotation.
    staging_done: Vec<Option<CudaEvent>>,
    /// Next staging slot to hand out.
    staging_next: usize,
    /// Where each zone slot is and how big it is.
    ///
    /// **Not one size.** A homed slot is exactly its layer's image; only the
    /// floating cell is sized for the largest streamable layer, because only it
    /// has to hold more than one thing. See [`super::zone`].
    slots: Vec<LayerPlacement>,
    /// `slot → view`, rebuilt whenever a slot's tenant changes.
    ///
    /// Behind an `Arc` so [`Self::ensure`] can hand the caller a handle rather
    /// than a borrow. A borrow would be tied to the `&mut self` that `ensure`
    /// takes, which makes the documented per-layer protocol — `ensure` then
    /// `prefetch` — fail to compile: the forward would be holding a reference
    /// into the very structure the next call mutates. The `Arc` also states the
    /// lifetime honestly, because `prefetch` may evict the slot the handle came
    /// from, and the handle must keep the assembled matmuls alive until the
    /// layer's compute has been issued.
    views: Vec<Option<Arc<T>>>,
    /// Turns a slot's views into what the forward reads. Called once per
    /// tenancy — when a layer lands in a slot, not when it is read.
    assemble: A,
    /// `layer → fence`, recorded after that layer's copies.
    fences: Vec<Option<CudaEvent>>,
    /// The planner's buffers, held so a per-layer plan allocates nothing.
    plan_scratch: PlanScratch,
    plan_ops: Vec<LoadOp>,
    stats: LayerCacheStats,
}

impl<T, A: SlotAssembler<T>> std::fmt::Debug for LayerCache<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerCache")
            .field("layers", &self.images.len())
            .field("slots", &self.slots.len())
            .field("warm", &self.warm.num_slots())
            .field("stats", &self.stats)
            .finish()
    }
}

mod cache_ops;
#[cfg(test)]
mod tests_streaming;
