//! Startup: fill the warm and hot tiers from the model pack's expert section.
//!
//! Every VRAM upload whose source is not already pinned — a pageable warm slot,
//! a pack read — goes through the pad's slots ([`StartupRing`]): the pad is
//! allocated before the warm tier and is idle until the stager takes it over,
//! so the fill borrows it as its landing ring.

use super::cache::ExpertCacheInner;
use super::pack::{ExpertPack, PackRead, RecordLayout};
use super::pad::Pad;
use super::pinned::{ExpertResidency, LayerGeometry};
use super::slot_image::build_slot_from_record_on_stream;
use super::types::ExpertSlot;
use super::warm_tier::WarmTier;
use candle::vram::{available_physical_ram, host_pinned_bytes};
use candle::Result;
use cudarc::driver::CudaEvent;

/// Everything the startup fill writes into, gathered so the fill takes one
/// argument for it instead of seven.
pub(crate) struct StartupTargets<'a> {
    pub inner: &'a mut ExpertCacheInner,
    pub warm: &'a mut WarmTier,
    pub residency: &'a mut [Vec<ExpertResidency>],
    /// Warm slot `i` holds `membership[i]`, decided once by the stratified draw.
    pub membership: &'a [(usize, usize)],
    pub geoms: &'a [LayerGeometry],
    pub layouts: &'a [RecordLayout],
    pub stride: usize,
}

/// Units the startup reports **after** the last expert, so the bar keeps moving
/// through the work that follows the fill: seeding the residency gauge, whose
/// caller reports it as it finishes.
pub(crate) const PACK_TAIL_STEPS: usize = 1;

/// The pad's slots as a ring of landing buffers for the startup fill.
///
/// The upload out of a buffer is asynchronous, so each buffer carries the event
/// of the upload it last fed and is waited on before it is written again — a
/// host wait, at startup, where nothing else is running.
pub(crate) struct StartupRing<'a> {
    pad: &'a Pad,
    events: Vec<Option<CudaEvent>>,
    next: usize,
}

impl<'a> StartupRing<'a> {
    pub(crate) fn new(pad: &'a Pad) -> Self {
        Self {
            pad,
            events: (0..pad.num_slots()).map(|_| None).collect(),
            next: 0,
        }
    }

    /// The next buffer, once the upload it last fed has landed.
    fn acquire(&mut self) -> Result<usize> {
        let idx = self.next;
        self.next = (self.next + 1) % self.events.len();
        if let Some(event) = self.events[idx].take() {
            event.synchronize().map_err(candle::Error::wrap)?;
        }
        Ok(idx)
    }

    fn buffer(&mut self, idx: usize) -> &mut [u8] {
        // SAFETY: `acquire` waited out the last upload from this slot, and the
        // ring is the pad's only user until the stager takes it over.
        unsafe { self.pad.slot_mut(idx) }
    }

    /// Record that buffer `idx` is the source of an upload that has not landed.
    fn publish(&mut self, idx: usize, event: CudaEvent) {
        self.events[idx] = Some(event);
    }
}

/// VRAM free, host RAM available, and host bytes pinned by this process, read
/// at the moment an upload failed — the three things a
/// `CUDA_ERROR_OUT_OF_MEMORY` on an upload can be about.
fn memory_state(cuda_dev: &candle::CudaDevice) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    let vram = match cuda_dev.mem_get_info() {
        Ok((free, total)) => format!(
            "VRAM free {:.2} of {:.2} GiB",
            free as f64 / GIB,
            total as f64 / GIB
        ),
        Err(e) => format!("VRAM unreadable: {e}"),
    };
    let available = available_physical_ram().map_or("unknown".to_string(), |b| {
        format!("{:.2} GiB", b as f64 / GIB)
    });
    format!(
        "{vram}, host RAM available {available}, host pinned {:.2} GiB",
        host_pinned_bytes() as f64 / GIB
    )
}

/// Upload the record in ring buffer `idx` into `slot_base`, and publish the
/// buffer behind the upload.
///
/// # Safety
///
/// `slot_base` names a slot the zone handed out and has not reclaimed.
unsafe fn upload_from_ring(
    ring: &mut StartupRing<'_>,
    idx: usize,
    layout: RecordLayout,
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    slot_base: u64,
) -> Result<ExpertSlot> {
    let stream = cuda_dev.cuda_stream();
    let slot = build_slot_from_record_on_stream(
        ring.buffer(idx),
        layout,
        geom,
        cuda_dev,
        &stream,
        slot_base,
    )?;
    ring.publish(idx, stream.record_event(None).map_err(candle::Error::wrap)?);
    Ok(slot)
}

/// Fill both resident tiers from the expert section.
///
/// Only the experts that land somewhere are read — the rest stay on disk until
/// something asks for them. VRAM fills in layer order, so the permanently
/// resident leading layers (`cache::PINNED_LAYERS`) take the first slots: the
/// zone's floor prices them, so they always fit.
pub(crate) fn startup_from_pack(
    t: StartupTargets<'_>,
    pack: &ExpertPack,
    ring: &mut StartupRing<'_>,
    num_moe_layers: usize,
    num_experts: usize,
    cuda_dev: &candle::CudaDevice,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<()> {
    if num_moe_layers == 0 || num_experts == 0 {
        return Ok(());
    }
    let total_experts = num_moe_layers * num_experts;
    let t0 = std::time::Instant::now();

    // ── Warm tier: every membership record at once, at full queue depth ──
    //
    // The tier's slots — pinned and pageable alike — are cut to the pack's
    // stride and sector-aligned, so each read lands in its final home with
    // nothing in between.
    if !t.membership.is_empty() {
        let stride = t.stride;
        let reads: Vec<PackRead<'_>> = t
            .membership
            .iter()
            .zip(t.warm.slots_mut(t.membership.len(), stride))
            .map(|(&(layer, expert), dest)| PackRead {
                layer,
                expert,
                dest,
            })
            .collect();
        pack.read_many(reads)?;
        for (slot, &(layer, expert)) in t.membership.iter().enumerate() {
            t.residency[layer][expert].ram = Some(slot);
        }
    }
    tracing::info!(
        target: "candle_transformers::expert_lre",
        warm_slots = t.membership.len(),
        secs = t0.elapsed().as_secs_f64(),
        "startup: warm tier filled from the pack"
    );

    // ── Hot tier: fill VRAM in layer order, from warm where possible ──
    let stream = cuda_dev.cuda_stream();
    let mut vram_count = 0usize;
    let mut cold_reads = 0usize;
    // The indices address four parallel collections and identify the expert in
    // the progress callback and the slot install, so they are the subject here
    // rather than an artefact of iterating one of them.
    #[allow(clippy::needless_range_loop)]
    'fill: for moe_idx in 0..num_moe_layers {
        let geom = &t.geoms[moe_idx];
        let layout = t.layouts[moe_idx];
        for expert_idx in 0..num_experts {
            let Some(slot_idx) = t.inner.take_free() else {
                break 'fill;
            };
            let slot_base = t.inner.slot_base(slot_idx);
            // Names the upload that failed: which expert, which source, how far
            // the fill had got, and the memory state at the moment it failed.
            let at = |source: &'static str| {
                move |e: candle::Error| {
                    e.context(format!(
                        "startup fill: L{moe_idx}E{expert_idx} from {source} into VRAM slot \
                         {slot_idx} ({vram_count} resident so far; {})",
                        memory_state(cuda_dev)
                    ))
                }
            };
            // SAFETY (every arm): `slot_idx` was just handed out by the zone and
            // is not reclaimed while this runs.
            let slot = match t.residency[moe_idx][expert_idx].ram {
                // A pinned warm slot is written once and never again, so it is a
                // source no later write can race — no event, no wait.
                Some(warm_slot) if t.warm.is_pinned(warm_slot) => unsafe {
                    build_slot_from_record_on_stream(
                        t.warm.slot_ref(warm_slot, t.stride),
                        layout,
                        geom,
                        cuda_dev,
                        &stream,
                        slot_base,
                    )
                    .map_err(at("a pinned warm slot"))?
                },
                // A pageable one is never an upload source: through the ring.
                Some(warm_slot) => {
                    let idx = ring.acquire()?;
                    ring.buffer(idx)
                        .copy_from_slice(t.warm.slot_ref(warm_slot, t.stride));
                    unsafe { upload_from_ring(ring, idx, layout, geom, cuda_dev, slot_base) }
                        .map_err(at("a pageable warm slot"))?
                }
                None => {
                    let idx = ring.acquire()?;
                    pack.read_into(moe_idx, expert_idx, ring.buffer(idx))?;
                    cold_reads += 1;
                    unsafe { upload_from_ring(ring, idx, layout, geom, cuda_dev, slot_base) }
                        .map_err(at("the pack"))?
                }
            };
            t.inner.install(slot_idx, moe_idx, expert_idx, slot);
            t.residency[moe_idx][expert_idx].vram = Some(slot_idx);
            vram_count += 1;
            if let Some(cb) = progress {
                cb(
                    moe_idx * num_experts + expert_idx + 1,
                    total_experts + PACK_TAIL_STEPS,
                );
            }
        }
    }
    // The uploads above are asynchronous; the resident tier must be complete
    // before the pipeline takes it over.
    stream.synchronize().map_err(candle::Error::wrap)?;
    // The fill stops as soon as VRAM is full, so the remaining experts never
    // reach the progress callback. Land it on the experts, leaving the tail.
    if let Some(cb) = progress {
        cb(total_experts, total_experts + PACK_TAIL_STEPS);
    }

    tracing::info!(
        target: "candle_transformers::expert_lre",
        secs = t0.elapsed().as_secs_f64(),
        vram_count,
        cold_reads,
        warm_slots = t.membership.len(),
        warm_gib = t.warm.total_bytes() as f64 / 1e9,
        pack = %pack.path().display(),
        "startup: filled from the pack"
    );
    Ok(())
}
