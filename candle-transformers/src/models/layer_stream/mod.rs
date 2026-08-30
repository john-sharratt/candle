//! Dense layer streaming — VRAM ↔ RAM ↔ NVMe, by layer.
//!
//! A dense checkpoint larger than the card is held as a working set of resident
//! layers with the rest streamed in behind the wave, the same way
//! [`expert_lre`](crate::models::expert_lre) holds a routed one. The design is
//! `docs/qwen38_layer_streaming.md`; this module is its implementation.
//!
//! ## Why it is not the expert cache
//!
//! The machinery is shared — equal-sized slots filled from the right
//! (`candle_nn::kv_cache::WeightZone`), a repacked cold tier, a pinned warm
//! tier, a fence-per-layer prefetch — but one property does not carry over:
//!
//! > **An expert cache has a hit rate. A layer cache does not.**
//!
//! Routing touches a subset of experts per layer, so residency converts into
//! avoided traffic and it is worth scoring what to keep. Every dense layer is
//! needed on every forward, so there is no subset to be lucky about and nothing
//! to score. Streamed bytes per forward are `total − resident`, and no policy
//! moves that floor.
//!
//! What follows from having no hit rate is that the policy is *simpler*, not
//! harder. There is no transition matrix, no access frequency, no eviction key.
//! The wave's position is the whole policy:
//!
//! * layers behind the wave are free — this forward will not read them again;
//! * layers ahead are protected — evicting one guarantees a stall;
//! * `L+1` and `L+2` are always in flight and joined at need;
//! * beyond that, load opportunistically into slots freed behind the wave,
//!   nearest-first, never at the cost of a layer due sooner.
//!
//! ## The ordering that comes for free
//!
//! Layer *i* lives in slot *i*. The weight zone's addresses **descend** as the
//! index rises, so layer 0 is the rightmost slot and layer 63 the leftmost —
//! and retraction eats the highest indices, which are the highest layer
//! numbers, which are the layers the wave reaches last.
//!
//! Eviction order and maximum prefetch lead time are therefore the same
//! ordering, with nothing to compute: the layer thrown out is always the one
//! there is the most time to fetch back. That is a property of the zone's
//! existing geometry rather than anything this module arranges.
//!
//! ## Module structure
//!
//! | File | Contents |
//! |------|----------|
//! | [`descriptor`] | The byte layout of a layer inside a slot — the one source of offsets |
//! | `pack` | The cold tier: a repacked record for every layer outside the pinned head |
//! | [`residency`] | Which layer is in which slot, and what to load next |
//! | [`warm`] | Which layers the pinned host tier holds |
//! | `view` | Wrapping a slot's bytes as the layer's matmuls, without moving them |
//! | `cache` | Residency, the tiers under it, and the transfers between |
//! | [`build`] | Reading a loaded layer's geometry, and the streaming pack build |

#[cfg(feature = "cuda")]
pub mod assemble;
#[cfg(feature = "cuda")]
pub mod boundary;
pub mod build;
#[cfg(feature = "cuda")]
pub mod cache;
pub mod descriptor;
pub mod order;
/// The cold tier is read by the CUDA cache and by nothing else, but its own
/// tests are pure file I/O and want to run on a machine with no GPU — so it
/// stays compiled everywhere and simply has no consumer off CUDA.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) mod pack;
pub mod residency;
#[cfg(feature = "cuda")]
pub mod view;
pub mod warm;
pub mod zone;

#[cfg(feature = "cuda")]
pub use boundary::growth_tally;
#[cfg(feature = "cuda")]
pub use build::LoadedLayer;
#[cfg(feature = "cuda")]
pub use cache::{pack_path_for, LayerCache, LayerCacheStats, COMMITTED_DEPTH};
pub use descriptor::{
    layer_image, slot_bytes_for_layers, FfnForm, ImageError, LayerImage, LayerTensor, MixKind,
    Placement, Projection, PROJECTION_ALIGN,
};
pub use order::{eviction_order, protection_order};
pub use pack::PackIdentity;
pub use residency::{LayerResidency, LoadOp, PlanScratch, Residence};
#[cfg(feature = "cuda")]
pub use view::{build_layer_view, StreamedLayer};
pub use warm::{warm_membership, warm_slots_for};
pub use zone::{plan_zone, LayerPlacement, ZoneError, ZonePlan};
