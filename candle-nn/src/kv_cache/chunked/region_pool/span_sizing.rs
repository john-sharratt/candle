//! Computes how large the reservation should be from what the card has free.
//!
//! The span is what the card reports as usable, less the cushion the CUDA pool
//! keeps and the budget for graph executables, which the driver allocates
//! outside the span. The dense weights are already resident when this runs and
//! are inside that figure, so they are not subtracted again. The layout at rest
//! is the persistence block, then the dense weights, then regions up to the
//! weight floor. The transient tier is absent, because it occupies nothing
//! between forwards. A governor that fails to answer is an error, since a
//! fallback there would silently size the span at the test constant.

use super::*;

/// The span's target size: everything left, less the cushion the CUDA pool keeps
/// and what else is planned outside the span (`graph_reserve`): the graph
/// executables' budget, and the model's runtime pool reserve.
///
/// **The graph budget is planned here because it cannot be placed here.** The
/// driver allocates an executable's memory itself, so it comes from outside the
/// span whatever the span's size; left unplanned, the executables took whatever
/// headroom the reservation happened to leave, until an instantiation found none
/// and failed mid-forward. Leaving exactly `graph_reserve` outside the span —
/// the figure the executable library is held to (`graph_budget_bytes`) — makes
/// it a tenant of the card's plan beside the pool's cushion.
///
/// With a governor — every production load installs one — `usable` is what
/// remains of the balloon-measured capacity `C` right now. The dense weights are
/// already resident when this runs, so they are already inside `usable`'s
/// accounting and **nothing further is subtracted for them** (see
/// [`span_from`]).
///
/// Without a governor, [`TEST_SPAN_BYTES`].
///
/// **A governor that fails to answer is an error, not a fallback.** These two
/// cases used to share one `_` arm, so a transient `usable()` failure on a
/// production card sized the whole span at the 2 GiB test constant and said
/// nothing — the reservation would come up a quarter of its intended size and
/// every downstream number would be quietly wrong. Absence of a governor is a
/// test binary; a governor that errors is a fault.
pub(super) fn span_target(ordinal: usize, graph_reserve: usize) -> Result<usize> {
    let Some(gov) = candle::vram::get(ordinal) else {
        return Ok(TEST_SPAN_BYTES);
    };
    let usable = gov.usable().map_err(|e| {
        candle::Error::Msg(format!(
            "reservation: the VRAM governor could not report usable bytes ({e}). \
             Refusing to size the span from the {TEST_SPAN_BYTES} B test constant \
             on a card that has a governor — that would silently claim a fraction \
             of the intended reservation."
        ))
    })?;
    Ok(span_from(
        usable as usize,
        gov.pool_cushion() as usize + graph_reserve,
    ))
}

/// The span, given what is left of `C` and what to leave outside it — the
/// pool's cushion and the graph budget.
///
/// **The dense weights are subtracted exactly once, and not here.** `usable()`
/// is `headroom.min(C − spent_by_us)` where `spent_by_us` is the drop in
/// headroom since `C` was measured — and the dense weights are resident by the
/// time this is called, so they are already inside that drop. Subtracting
/// `class_reserved(Weights)` on top would book the same bytes twice.
///
/// That is not a hypothetical. The load order now makes the weights both
/// resident *and* tallied at this moment, so the second subtraction looks like
/// prudence; it is the exact mistake this codebase has already made twice.
/// `balloon_headroom_abs` reserved the transient peak that `expert_budget` was
/// also reserving — 1,104 MiB no allocator was permitted to touch — and this
/// function's predecessor subtracted `scratch_margin` against a transient tier
/// that was then added back on top, "same bytes, two places, opposite signs".
///
/// Split out so the identity can be asserted without a governor.
pub(super) fn span_from(usable: usize, pool_reserve: usize) -> usize {
    usable.saturating_sub(pool_reserve)
}

/// How the span divides, once its true backed size is known.
///
/// **This is the layout at rest**, between forwards. Only the persistence block
/// is here; the wave transient tier occupies nothing until a forward places it
/// ([`place_transient`]), so the regions run straight to the weight floor:
///
/// ```text
/// | persist 64 MiB | dense weights | regions … | ← W → weight slots |
/// ```
///
/// The **dense block** holds the model's permanent tensors. It is variable-size
/// like the tier, but for the opposite reason and with the opposite discipline:
/// it grows only during load, by [`claim_dense`], and once the first region is
/// handed out it can never move again — every region address is an offset from
/// its right edge, so shifting it afterwards would relocate live arenas. That
/// rule is enforced rather than documented ([`claim_dense`] refuses once `next`
/// has advanced), which is what lets it be a bump pointer with no free list: the
/// weights are loaded once and released only when the process ends.
///
/// Sizing it needs no estimate, which is the point. The span is claimed **before
/// the model loads**, so the weights are allocated out of it as they arrive and
/// the region count is derived from whatever is left — where previously the span
/// was sized from the memory the weights had already taken from the pool, and so
/// could not contain them.
///
/// The tier is deliberately absent. It is the one variable-size block that
/// changes size *at runtime* — priced per wave rather than at the widest wave —
/// and it is safe to be variable precisely because it vanishes here: at the
/// moment its extent changes it holds nothing, so a resize leaves no hole and
/// moves no data.
///
/// That is also why it belongs between the regions and the weights rather than
/// at a fixed address beside the persistence block. A variable-size block at a
/// fixed address is a choice between two failures: leave the region base fixed
/// and every resize strands a hole (fragmentation), or pack the regions after it
/// and every resize relocates every region (thrashing). Only the position
/// adjacent to the moving boundary disturbs nothing.
///
/// A span that came up short costs **regions** and never the persistence block:
/// it is positioned from the left and the region count is what absorbs the
/// difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) region_base: u64,
    pub(super) total: usize,
    /// First byte of the dense weight block — fixed at the persistence block's
    /// right edge, and the one address in the layout that never moves.
    pub(super) dense_base: u64,
    pub(super) transient_end: u64,
    /// The weight floor as clamped into the span — what the caller asked for,
    /// corrected. Returned rather than recomputed so nothing downstream has to
    /// repeat the clamp and risk disagreeing about it.
    pub(super) weight_floor: u64,
    pub(super) slack: usize,
}

/// Divide a span of `span_bytes` at `span_base` whose fixed left block is
/// `persist` bytes, with `dense_bytes` of model weights after that block and
/// the weight side occupying `[weight_floor, span_end)`.
///
/// The dense block is rounded up to a whole region so the region grid stays
/// aligned to the reservation's granule. That wastes under one region and buys
/// an address for every region that is `region_base + i · REGION_BYTES` with no
/// second alignment rule to keep in step.
pub(super) fn layout_span(
    span_base: u64,
    persist: usize,
    span_bytes: usize,
    dense_bytes: usize,
    weight_floor: u64,
) -> Layout {
    let span_end = span_base + span_bytes as u64;
    let dense_base = span_base + persist as u64;
    let dense_span = dense_bytes.next_multiple_of(REGION_BYTES);
    // Saturating, and clamped into the span: a dense block wider than the span
    // is refused by `claim_dense` before it can be stored, but the arithmetic
    // here must stay sane for the refusal path to be able to *compute* the
    // layout it is refusing.
    let region_base = dense_base.saturating_add(dense_span as u64).min(span_end);
    let transient_end = region_base;
    // Clamp rather than trust: the floor arrives from arithmetic done a crate
    // away, and a wrapped subtraction here would produce an address that looks
    // plausible. `.min(span_end)` keeps the lower bound meaningful on a span too
    // small to hold the fixed blocks; `RegionPool::create` refuses that span
    // outright, so this only has to stay arithmetically sane.
    let weight_floor = weight_floor.clamp(region_base.min(span_end), span_end);
    let usable_for_regions = weight_floor.saturating_sub(region_base) as usize;
    let total = usable_for_regions / REGION_BYTES;
    Layout {
        region_base,
        total,
        dense_base,
        transient_end,
        weight_floor,
        slack: usable_for_regions - total * REGION_BYTES,
    }
}

/// Carve the persistence thread's staging block, out of the fixed left end.
///
/// Separate from [`place_transient`] because the block is: it is carved **once**
/// and never moves, where the wave tier is placed and released every forward.
/// The persistence thread's copy stream is not synchronised to a wave, so its
/// ranges can be live at any moment one begins — it is the one domain that
/// cannot be allowed to vanish, and therefore the one that needs a fixed
/// address.
pub(crate) fn carve_persist(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<u64> {
    with_pool(stream, |pool| {
        let len = bytes.div_ceil(REGION_BYTES) * REGION_BYTES;
        if pool.persist_carved + len > pool.persist_bytes {
            candle::bail!(
                "persistence staging block exhausted: {len} B on top of {} B carved \
                 exceeds the {} B block",
                pool.persist_carved,
                pool.persist_bytes,
            )
        }
        let base = pool.span_base + pool.persist_carved as u64;
        pool.persist_carved += len;
        Ok(base)
    })
}
