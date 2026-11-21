//! Samples the pool's geometry between forwards, so the figures describe the
//! pool rather than a forward half built.
//!
//! A composition splits the arena frontier into packed regions, span-tenant
//! regions, sparsity and holes, so an efficiency figure can say whether the loss
//! is air inside the arenas or free regions below the frontier. Samples are read
//! while no transient tier is placed, because a placed tier makes every region
//! above its base read as unavailable. Each sample is identified by the publish
//! it came from, so repeated reads of one publish are not taken as persistence.

use super::*;

/// One sample of the pool's geometry.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Geometry {
    pub(super) live: usize,
    pub(super) watermark: usize,
    pub(super) free: usize,
    pub(super) weight_mib: usize,
    /// A tier was standing when this was read, so `free` is the ceiling's answer
    /// rather than the pool's — see [`sample`].
    pub(super) mid_wave: bool,
}

impl Geometry {
    /// Free regions stranded below the arena frontier — the damage, in regions.
    pub(super) fn holes(&self) -> usize {
        self.watermark.saturating_sub(self.live)
    }
}

/// The VRAM composition below the frontier, **as one publish of the engine's own
/// memory report sees it**.
///
/// Every field is in regions except [`Self::denied_mib`], and together they
/// decompose the frontier: `frontier = packed + span + sparsity + holes`, where
/// sparsity is `arenas - packed` and holes is `frontier - live`. Kept as a struct
/// rather than a tuple because the figures are only interpretable together — an
/// efficiency percentage on its own cannot say whether the loss is air inside the
/// arenas, which a pack removes, or free regions stranded below the frontier, which
/// the next claim takes by itself.
///
/// `span` is the term that is not a loss: whole regions a span tenant holds — a
/// sequence's recurrent state store, the provenance gallery. Without it the figure is
/// unreadable on any model that has one. Qwen3.8-Flash-Next, whose DeltaNet layers
/// keep their state there, read 4% efficiency with the KV pools packed to within two
/// arenas of perfect, because a few hundred regions of live recurrent state were being
/// charged to compaction as fragmentation.
///
/// All of it comes from the report and none from a second `region_stats` call, so the
/// numerator and the denominator are one moment. See [`Composition::frontier`].
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Composition {
    /// What a perfect pack of the KV pools would return.
    pub(super) denied_mib: usize,
    /// `(packed + span) / frontier`, as a percentage.
    pub(super) eff: usize,
    /// Arenas the GPU KV pools hold.
    pub(super) arenas: usize,
    /// Arenas those pools would need if packed.
    pub(super) packed: usize,
    /// The arena frontier at the same instant as the rows above.
    pub(super) frontier: usize,
    /// Regions held by any tenant at that instant.
    pub(super) live: usize,
    /// Of `live`, the regions a span tenant holds. In use, and not packable.
    pub(super) span: usize,
    /// Of `live`, the regions `KvHead` record arenas hold. In use, not packable, and in no
    /// size-class row — so absent from `packed` and `arenas` both.
    pub(super) record: usize,
    /// Everything below the frontier that is holding nothing, in MiB — sparsity plus
    /// holes. The absolute form of [`Self::eff`], and the unit the weight side
    /// actually loses.
    pub(super) loss_mib: usize,
    /// When the engine published this report, in milliseconds since the epoch.
    ///
    /// **The identity of the observation, not decoration.** The report is published on
    /// the scheduler's telemetry cadence and this harness samples faster than that, so
    /// consecutive samples routinely read the *same* publish. Anything asking whether a
    /// figure persisted has to compare distinct publishes, or it is comparing one
    /// observation with itself and finding, unsurprisingly, that it agrees.
    pub(super) captured_ms: u64,
}

pub(super) fn sample_now(device: &Device) -> Option<Geometry> {
    let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
        return None;
    };
    region_stats(gpu_id).map(|s| Geometry {
        live: s.live,
        watermark: s.live_watermark,
        free: s.free,
        weight_mib: s.weight_bytes >> 20,
        mid_wave: s.transient_bytes > 0,
    })
}

/// A reading taken **between forwards**, retrying briefly for one.
///
/// Sampling indiscriminately reports nonsense for `free`: while a forward is open
/// the transient tier is placed, and the region ceiling then puts every region
/// above the tier's base out of reach, so `free` reads 0 no matter how empty the
/// pool is. A saturation test fed those samples declares victory on the first
/// forward it happens to catch — which is exactly what the first version of this
/// harness did, reporting "saturated after 3.0s" against a pool with 1,700 free
/// regions.
///
/// `transient_bytes > 0` is the discriminator: non-zero means a tier stands, so
/// the reading is mid-wave. Falls back to the last reading if no quiet moment
/// appears, flagged so the caller can say so rather than quoting it as fact.
pub(super) fn sample(device: &Device) -> Option<Geometry> {
    let mut last = None;
    for _ in 0..40 {
        let g = sample_now(device)?;
        if !g.mid_wave {
            return Some(g);
        }
        last = Some(g);
        std::thread::sleep(Duration::from_millis(25));
    }
    last
}
