//! Where a layer's weights come from.
//!
//! The forward asks for layer `li` and gets a [`QuantLayer`]. This is the one
//! place that knows whether that layer was sitting in VRAM the whole time or
//! was streamed into a weight-zone slot moments ago.
//!
//! # Why there are two variants and why that is not a fits/does-not-fit branch
//!
//! `docs/qwen38_layer_streaming.md` §7 forbids deciding at load time whether a
//! model fits and taking a different path if it does not — that is two code
//! paths for one job, and the streaming half would be exercised only by the
//! largest checkpoint anyone happens to run. **Every dense checkpoint streams**,
//! and "it fits" is the degenerate case of one mechanism: capacity equals the
//! layer count, nothing is ever evicted, and no byte moves after load.
//!
//! The split here is a different question with a different answer. A *routed*
//! checkpoint's bulk is its experts, which `expert_lre` already streams; its
//! per-layer projections are a rounding error beside them, and there is nothing
//! to gain by moving those through slots as well. So the variant is chosen by
//! what kind of model this is, not by whether it happened to fit — and the
//! forward cannot tell the difference either way, because it only ever calls
//! [`LayerStore::ensure`].
//!
//! # The residue is resident in both
//!
//! Norms, the DeltaNet recurrence's F32 constants, and the two sub-tile gates
//! never enter a slot (see `layer_stream::descriptor`). Consumers that want
//! only those — the speculative verify replay, which runs at accept time when
//! the layer may well have been evicted, and the activation-dtype change, which
//! runs at session setup — go through [`LayerStore::residue`] and never touch
//! the streaming machinery at all. That is not an optimisation: `ensure` at
//! accept time would fetch a whole layer over PCIe to read four small
//! constants that were in VRAM the entire time.

use std::sync::Arc;

use candle::Result;

use super::quantized_weights::{QuantLayer, ResidentResidue};

#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use super::layer_loader::QwenLayerCache;
#[cfg(feature = "cuda")]
use crate::models::layer_stream::{boundary, LayerCacheStats};

/// The layers of a loaded model, however they are held.
pub enum LayerStore {
    /// Every layer resident for the life of the process.
    ///
    /// Routed checkpoints, whose weight is in the experts, and the unit tests
    /// that build a model without a checkpoint on disk to pack.
    Resident(Vec<Arc<QuantLayer>>),
    /// Dense layers as tenants of the weight zone's slots.
    #[cfg(feature = "cuda")]
    Streamed(StreamedLayers),
}

/// A dense model's layers, held as slot tenants.
#[cfg(feature = "cuda")]
pub struct StreamedLayers {
    /// Behind a `Mutex` because the forward takes `&QuantModel` and a model is
    /// shared across threads, while advancing the wave and issuing transfers is
    /// inherently mutation. Uncontended in practice — one wave is open at a
    /// time — so this is interior mutability rather than synchronisation.
    ///
    /// And behind an `Arc` because the **ground broker** holds a `Weak` to it.
    /// `region_pool::buy_ground` is a process-global hook, not a call on the
    /// model, so the seller has to be reachable from a `'static` closure that
    /// must not keep the model alive — exactly the shape
    /// `expert_loader` registers for a routed checkpoint.
    cache: Arc<Mutex<QwenLayerCache>>,
    /// Shared with the cache's assembler, which clones handles out of it into
    /// every tenancy.
    residues: Arc<Vec<ResidentResidue>>,
}

#[cfg(feature = "cuda")]
impl StreamedLayers {
    pub fn new(cache: Arc<Mutex<QwenLayerCache>>, residues: Arc<Vec<ResidentResidue>>) -> Self {
        Self { cache, residues }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, QwenLayerCache>> {
        self.cache
            .lock()
            .map_err(|_| candle::Error::Msg("layer store: the layer cache lock is poisoned".into()))
    }
}

/// Sell `regions` of weight-zone ground to the KV side, through a shared handle.
///
/// The ground broker's body, and [`LayerStore::request_kv_ground`]'s. Free
/// rather than a method because the broker is a `'static` closure over a `Weak`
/// and has no `LayerStore` to call.
#[cfg(feature = "cuda")]
pub fn sell_ground(cache: &Mutex<QwenLayerCache>, regions: usize) -> u64 {
    let sell = || -> Result<u64> {
        let mut c = cache.lock().map_err(|_| {
            candle::Error::Msg("layer store: the layer cache lock is poisoned".into())
        })?;
        boundary::concede_kv_ground(&mut c, regions)
    };
    match sell() {
        Ok(bytes) => bytes,
        Err(e) => {
            // The refusal that lands here is a wave generation still open on the
            // span. Warned rather than propagated: the caller's next move is to
            // retry or to fail the wave on its own terms, and neither is
            // improved by this becoming the error it reports.
            tracing::warn!(
                target: "candle_transformers::layer_stream",
                "layer zone could not concede ground: {e}"
            );
            0
        }
    }
}

impl LayerStore {
    /// How many layers the trunk has.
    pub fn len(&self) -> usize {
        match self {
            Self::Resident(v) => v.len(),
            #[cfg(feature = "cuda")]
            Self::Streamed(s) => s.residues.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The layer's weights, moving the wave to it and joining any transfer of
    /// it that is still in flight.
    ///
    /// The only place the forward can block on the streaming subsystem, and on
    /// a resident store it is an `Arc` refcount bump.
    pub fn ensure(&self, layer: usize) -> Result<Arc<QuantLayer>> {
        match self {
            Self::Resident(v) => v.get(layer).map(Arc::clone).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "layer store: L{layer} is past the trunk's {} layers",
                    v.len()
                ))
            }),
            #[cfg(feature = "cuda")]
            Self::Streamed(s) => s.lock()?.ensure(layer),
        }
    }

    /// Issue the loads the wave's position asks for.
    ///
    /// Called after the layer's compute has been issued, so the copies overlap
    /// it rather than serialising in front of it.
    pub fn prefetch(&self) -> Result<()> {
        match self {
            Self::Resident(_) => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Streamed(s) => s.lock()?.prefetch(),
        }
    }

    /// The part of a layer that never enters a slot — norms, the recurrence's
    /// constants, the sub-tile gates.
    ///
    /// Cheap in both variants: every field is a handle, so this is a few
    /// refcount bumps and no device traffic.
    pub fn residue(&self, layer: usize) -> Result<ResidentResidue> {
        match self {
            Self::Resident(v) => v
                .get(layer)
                .map(|l| l.residue())
                .ok_or_else(|| candle::Error::Msg(format!("layer store: no layer {layer}"))),
            #[cfg(feature = "cuda")]
            Self::Streamed(s) => s
                .residues
                .get(layer)
                .cloned()
                .ok_or_else(|| candle::Error::Msg(format!("layer store: no residue {layer}"))),
        }
    }

    /// The streaming counters, or `None` when nothing streams.
    #[cfg(feature = "cuda")]
    pub fn stats(&self) -> Result<Option<LayerCacheStats>> {
        match self {
            Self::Resident(_) => Ok(None),
            Self::Streamed(s) => Ok(Some(s.lock()?.stats())),
        }
    }

    /// Surrender layer slots so the KV side can claim `regions` more regions,
    /// answering with the bytes conceded.
    ///
    /// **Between forwards only** — see [`crate::models::layer_stream::boundary`].
    /// A resident store concedes nothing: its layers are in the dense block,
    /// which is frozen and has no give.
    #[cfg(feature = "cuda")]
    pub fn request_kv_ground(&self, regions: usize) -> u64 {
        match self {
            Self::Resident(_) => 0,
            Self::Streamed(s) => sell_ground(&s.cache, regions),
        }
    }

    /// Take back KV ground standing free. **Between forwards only.**
    #[cfg(feature = "cuda")]
    pub fn reclaim_spare_ground(&self) {
        let Self::Streamed(s) = self else {
            return;
        };
        let reclaim = || -> Result<u64> {
            let mut c = s.lock()?;
            boundary::reclaim_spare_ground(&mut c)
        };
        if let Err(e) = reclaim() {
            tracing::warn!(
                target: "candle_transformers::layer_stream",
                "layer zone could not take spare KV ground: {e}"
            );
        }
    }
}

impl std::fmt::Debug for LayerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resident(v) => f
                .debug_struct("LayerStore::Resident")
                .field("layers", &v.len())
                .finish(),
            #[cfg(feature = "cuda")]
            Self::Streamed(s) => f
                .debug_struct("LayerStore::Streamed")
                .field("layers", &s.residues.len())
                .finish(),
        }
    }
}
