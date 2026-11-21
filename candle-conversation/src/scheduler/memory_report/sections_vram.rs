//! Describes the card's VRAM: the held ground, the weights, the gallery and the
//! whole-card reconciliation.
//!
//! The reservation is the budget, and memory allocated outside it competes with
//! it, so `outside_span_bytes` is the figure to drive to zero. Each named tenant
//! is reported as a part of the total that contains it, never as an addend,
//! because the dense weights and the pool would otherwise be counted twice. The
//! unaccounted residual is measured against the driver's baseline taken at
//! reservation, so it reads near zero unless something unknown holds memory.

use super::*;

/// The reservation's held ground: its totals, and who holds it.
///
/// [`AccountingSection::inside_held_bytes`] says how full the ground is; this
/// says by whom. A buffer the ground refuses goes to the CUDA pool and, on a
/// full card, to the driver — so when a forward dies of
/// `CUDA_ERROR_OUT_OF_MEMORY`, the holders listed here are what filled it and
/// the `declines` are who was turned away.
#[derive(Debug, Clone, Serialize)]
pub struct HeldSection {
    /// Bytes taken and not yet reusable, padding and fenced releases included.
    pub used_bytes: u64,
    /// The part of `used_bytes` given back but still behind its fence.
    pub pending_bytes: u64,
    pub peak_bytes: u64,
    pub capacity_bytes: u64,
    /// The widest buffer the ground could hand out now.
    pub largest_free_bytes: u64,
    /// Every holder that has taken or been refused anything, largest first.
    pub sites: Vec<HeldSiteRow>,
}

/// One holder of held ground, named by the code that asked for it.
#[derive(Debug, Clone, Serialize)]
pub struct HeldSiteRow {
    /// `file:line:column`.
    pub site: String,
    pub live_bytes: u64,
    pub live_count: usize,
    pub peak_bytes: u64,
    pub takes: u64,
    /// Requests the ground could not hold — each one went to the CUDA pool.
    pub declines: u64,
    pub declined_bytes: u64,
    pub largest_decline_bytes: u64,
}

/// Model weights resident in VRAM, split into the two parts that behave
/// differently: the dense tensors, which are permanent, and the expert slots,
/// which page against host RAM.
///
/// Without this the dense half is invisible to every consumer of the report —
/// it sits outside the KV pool (so it is absent from `pool_used_bytes`) and its
/// [`AllocClass::Weights`] tally is never credited, so a whole-card accounting
/// could only reach it by subtraction.
#[derive(Debug, Clone, Serialize)]
pub struct WeightSection {
    /// Dense tensors: embeddings, attention projections, norms, output head.
    /// `None` when the model cannot report it (non-MoE or no expert cache).
    pub base_bytes: Option<u64>,
    /// Expert slots currently resident in VRAM — moves as experts page.
    pub resident_expert_bytes: Option<u64>,
    /// The weight zone's floor: ground the weight side must keep, and therefore
    /// ground **K/V may not plan against**. `None` off a device reservation.
    ///
    /// Reported because a consumer sizing a batch of new conversations has to
    /// subtract it and had no way to see it. The `repo_map` pool sized itself
    /// from the governor's whole capacity, so at 96 conversations its K/V grew
    /// to 48 GB and squeezed the weight zone from 53.5 GB to 10.9 GB — 19 GB
    /// under this floor. Admission then refused every prefill, correctly and
    /// permanently: residency was below the floor, and refusing prefills is not
    /// a thing that gives K/V back.
    pub floor_bytes: Option<u64>,
}

/// The provenance gallery arena's VRAM slabs.
///
/// The arena allocates 16 MiB device slabs directly and holds them for its
/// lifetime, outside both the KV pool and the governor's class tallies, so it
/// appears in no other field of this report. Its pages are evictable and rebuild
/// from the substrate on demand, which is why the relief sequence sheds them
/// before it touches model KV.
#[derive(Debug, Clone, Serialize)]
pub struct GallerySection {
    /// VRAM held by the arena's slabs right now.
    pub resident_bytes: u64,
    /// Turns currently paged into the arena.
    pub resident_turns: usize,
}

/// Device + pool + governor view of the card.
#[derive(Debug, Clone, Serialize)]
pub struct VramSection {
    /// `cuMemGetInfo` free/total — honest under TCC and Linux; under WDDM free
    /// is polluted by what the OS manages, and the governor's DXGI headroom is
    /// the residency signal.
    pub driver_free_bytes: u64,
    pub driver_total_bytes: u64,
    /// CUDA stream-ordered pool: live allocations vs OS-reserved footprint.
    /// `reserved - used` is the reuse gap — reusable for pool allocations that
    /// fit existing arenas, NOT usable for fresh arenas, and the first thing
    /// WDDM spills to host memory near the ceiling.
    pub pool_used_bytes: u64,
    pub pool_reserved_bytes: u64,
    /// Pool-aware availability (`init_free - pool_used - reserve`).
    pub budget_available_bytes: Option<u64>,
    /// Governor view — absent when no governor is installed (non-CUDA).
    pub governor: Option<GovernorSection>,
}

/// The VRAM governor's budget model.
#[derive(Debug, Clone, Serialize)]
pub struct GovernorSection {
    /// The device's driver model — `wddm`, `tcc` or `linux`
    /// (`docs/driver_model.md`); absent for a probe-only governor.
    pub driver_model: Option<&'static str>,
    /// The probe the readings below come from: `dxgi` under WDDM, `cuda` under
    /// TCC and Linux. The two measure headroom on different scales.
    pub probe: &'static str,
    /// Balloon-measured resident capacity `C`.
    pub capacity_bytes: u64,
    /// Live measured headroom (honest, excludes the pool reuse gap).
    pub headroom_bytes: u64,
    /// Cushion left outside the reservation for the CUDA pool.
    ///
    /// Replaces `kv_floor_bytes`, which reported the static KV reserve. There is
    /// no static reserve now — the span holds KV, transients and experts, and
    /// the boundary between the last two moves — so the weight side's own extent
    /// is what says where the partition currently sits, and that is reported
    /// with the arena occupancy rather than here.
    pub pool_cushion_bytes: u64,
    /// Loose per-class reserved tallies (reporting, not availability gates).
    pub reserved_weights_bytes: u64,
    pub reserved_expert_bytes: u64,
    pub reserved_scratch_bytes: u64,
    pub reserved_kv_bytes: u64,
}

/// Whole-card reconciliation: what this report can account for, and what it
/// cannot.
///
/// Every other section names one consumer. None of them adds up, and that is
/// how several GiB stayed invisible: the dense weights, the gallery arena's
/// slabs and the CUDA pool each sat outside the KV pool and outside each
/// other's tallies, so no reader could have summed them without knowing to look
/// in three places and a fourth for the total. All three are inside the
/// reservation now, and this section is where that is checked rather than
/// assumed.
///
/// [`Self::outside_span_bytes`] is the number that matters. The reservation is
/// the budget; memory allocated outside it competes with it, and on WDDM the
/// loser is paged to host RAM — measured at 3.7 GiB on the 3.6-35B, which cost
/// 17x on decode with nothing anywhere saying why. Driving it to zero is the
/// goal, and this is the meter for it.
#[derive(Debug, Clone, Serialize)]
pub struct AccountingSection {
    /// The VMM reservation: pinned, non-migratable, holds KV + experts +
    /// transients. `None` without a region pool.
    pub span_bytes: Option<u64>,
    /// Ours, but NOT in the span — the demotable set.
    ///
    /// This **is** [`Self::outside_pool_bytes`], because on this device every
    /// device allocation the engine makes outside the reservation is a CUDA pool
    /// allocation: `CudaDevice::{alloc, alloc_zeros, memcpy_stod}` all reach
    /// `CudaStream::alloc`, which calls `cuMemAllocAsync` whenever the context
    /// reports `has_async_alloc`. The pool's OS-reserved footprint therefore
    /// covers the whole outside set already.
    pub outside_span_bytes: u64,
    /// The CUDA async pool's OS-reserved footprint — the whole outside set.
    pub outside_pool_bytes: u64,
    // There is deliberately no `outside_dense_bytes`. It was wrong three times,
    // and the third is the instructive one.
    //
    // First it was an ADDEND to the pool, as though the weights sat beside it
    // rather than in it — inflating the demotable set by the size of the model.
    // Then a named PART of the pool, correct until the weights moved into the
    // span, at which point it reported a component larger than the whole
    // containing it. Then a SUBTRACTION, `total_dense - inside_dense`, which
    // looked principled and was not: `total_dense` sums the raw GGUF bytes the
    // loader read, `inside_dense` sums the repacked twins that stayed, and a
    // Q4_K source is not the size of its Q4_KO twin. Subtracting two different
    // populations produced a confident 135 MiB of weights that did not exist.
    //
    // What answers the question exactly is already here: [`Self::inside_dense_bytes`]
    // is what the allocator handed out, and [`Self::outside_pool_bytes`] is what
    // the driver says the pool holds. Right after load the second is the residual
    // weight footprint — measured at 32 MiB, against 1,952 before the weights
    // moved into the span. Neither is derived from the other.
    /// The gallery arena, which is **inside** the span: its slabs are claimed
    /// regions, the same 16 MiB unit a KV arena takes.
    ///
    /// Reported here beside the outside set rather than only in
    /// [`GallerySection`] because it used to be part of it — the slabs came
    /// from the CUDA pool and were, in the arena's own words, "never returned".
    /// A reader comparing two runs needs to see that this moved rather than
    /// vanished, and a regression that put it back outside would otherwise show
    /// up only as `outside_pool_bytes` quietly growing.
    pub inside_gallery_bytes: u64,
    /// The model's dense weights, **inside** the span — the dense block, locked
    /// at the end of load.
    ///
    /// The whole point of the reservation-before-load ordering: these bytes were
    /// the single largest thing outside the span, and being outside it meant
    /// WDDM could demote the model itself to host RAM. Inside, they are pinned
    /// device allocations the driver may not migrate.
    pub inside_dense_bytes: u64,
    /// Per-sequence recurrent state, also **inside** the span: each store's
    /// buffers are carved from claimed regions.
    ///
    /// Named for the same reason [`Self::inside_gallery_bytes`] is, and more
    /// urgently — it is the largest thing in the span that is neither KV nor
    /// weights (~126 MiB per sequence on a hybrid stack, several GiB across a
    /// wide wave) and it moves with wave width, so a run that grows it and a run
    /// that leaks it look identical in the span total alone.
    pub inside_recurrent_bytes: u64,
    /// The held ground's buffers, **inside** the span: the launch tables,
    /// resident headers, sampler packs and attention partials that outlive
    /// every forward. What the pool used to hold of them now shows up here, and
    /// a buffer the ground could not hold shows up in `outside_pool_bytes`.
    pub inside_held_bytes: u64,
    /// Bytes that went to the pool because their origin carried no wave ticket,
    /// **over the last wave**.
    ///
    /// A provenance break: something upstream produced a tensor with no wave
    /// backing and everything derived from it inherited the pool, so the fix is
    /// at that root — which may be many frames above whatever site a
    /// forbidden-allocation report names.
    ///
    /// Per-wave rather than cumulative, and the distinction is load-bearing. The
    /// lifetime total is dominated by declines that are *correct*: an op on the
    /// residual stream has nothing to inherit, because the residual crosses
    /// layers and belongs on the pool by design, and model loading has no wave at
    /// all. Read cumulatively this number is large, unfalsifiable, and says
    /// nothing. Scoped to a wave — where every allocation is supposed to inherit
    /// — a non-zero reading is a defect.
    pub decline_no_ticket_bytes: u64,
    /// Bytes that went to the pool because their operand's generation had
    /// closed, over the last wave — a value read after its phase, whose derived
    /// data has no phase left to live in.
    ///
    /// Reported beside [`Self::decline_no_ticket_bytes`] because the two land on
    /// the same `CudaDevice::alloc` and are otherwise indistinguishable. An arena
    /// with no room is neither: it fails the wave, naming the request.
    pub decline_closed_bytes: u64,
    /// Driver-reported in use across the whole card (`total - free`).
    pub device_in_use_bytes: u64,
    /// The **expert weight zone's extent** inside the span — `span_end −
    /// weight_floor`, the ground the weight side is permitted to hold.
    ///
    /// **Capacity, not occupancy.** It counts the zone's free slots too, so it
    /// is always ≥ [`WeightSection::resident_expert_bytes`], which is what is
    /// actually loaded. The two answer different questions and the report needs
    /// both: this one says how much of the span the weight side owns (and is
    /// therefore the tenant figure that belongs beside `inside_dense_bytes` and
    /// friends), while the resident figure says how much of that ground is in
    /// use. Naming it `inside_expert_bytes` and calling it "resident" — as the
    /// first version of this field did — puts two different quantities under one
    /// heading and silently over-counts occupancy by the zone's free capacity.
    ///
    /// This is also the tenant that *moves*: the elastic boundary concedes it to
    /// the KV side under pressure, so watching it fall across a run is how zone
    /// erosion becomes visible in the report instead of only in the log.
    pub expert_zone_bytes: u64,
    /// The wave transient tier, **inside** the span — this forward's activation
    /// ground, between the KV regions and the weight zone.
    ///
    /// Transient by construction (released between forwards), so a reading here
    /// is a point-in-time sample rather than a standing cost. Named because it
    /// is the fourth span tenant and its absence made the span's own arithmetic
    /// impossible to close.
    pub inside_transient_bytes: u64,
    /// Device bytes in use before the span was reserved — the CUDA context, the
    /// driver's working set, the module images, and any other process. Measured
    /// at reservation time, which is the only moment it stands alone (the
    /// reservation precedes the weight load).
    ///
    /// Published so [`Self::unaccounted_bytes`] can mean one thing instead of
    /// three.
    ///
    /// **`None` when it could not be measured**, which is a real state: the
    /// reservation runs whether or not a VRAM gauge was registered (it branches
    /// on exactly that), so on a gauge-less run there is no baseline to take. It
    /// is an `Option` rather than a `0` because the two readings demand opposite
    /// conclusions — a genuine zero means the card was empty before us and
    /// `unaccounted` is trustworthy, while an unmeasured one means
    /// `unaccounted` has silently reverted to including the whole CUDA context,
    /// which is the 1.4 GB of noise this field exists to remove. Reporting both
    /// as `0` hides that distinction at exactly the moment a reader is trying to
    /// decide whether a residual is a leak.
    pub driver_baseline_bytes: Option<u64>,
    /// `device_in_use - (span + outside + driver_baseline)` — **an allocation
    /// this report does not know about, and nothing else** — but only when
    /// [`Self::driver_baseline_bytes`] is `Some`. With `None` the baseline could
    /// not be subtracted and this reverts to the older, inflated meaning; read
    /// that field before reading this one.
    ///
    /// It used to fold in the driver baseline too, and read 1.4 GB on a 16 GB
    /// card while being almost entirely that constant. A number that large is
    /// unreadable as a signal: a genuine few-hundred-MB leak moves it by a
    /// fraction and nobody can tell. With the baseline named separately this is
    /// meant to sit near zero, and any material reading is a defect to chase.
    pub unaccounted_bytes: i64,
}
