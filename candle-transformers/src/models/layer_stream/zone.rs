//! Turning a byte budget into layer addresses.
//!
//! # Two tiers, because only one of them is a cache
//!
//! The weight zone used to be an array of equal cells, any of which could hold
//! any layer. That is what a *cache* needs, and it is the wrong shape here: on
//! this model 34 of 38 cells held a layer that never moved, and each of them
//! paid the difference between its tenant and the largest layer in the model —
//! `max` over layers rather than each layer's own size. Measured on the 27B that
//! is 202.8 MiB per cell against a 165.6 MiB mean, so roughly a fifth of the
//! zone was reserved for layers that were somewhere else.
//!
//! So the zone is split by what a layer actually does:
//!
//! ```text
//! span_end ─┐
//!           │  resident tier   layers at their own size, most protected first
//!           │                  no cell, no eviction, no dead space
//!           │  floating cell   one, sized for the largest streamable layer
//! floor  ───┘
//! ```
//!
//! The streaming machinery genuinely needs one cell: the layer under the wave
//! occupies it while the next missing layer's fetch waits for it to be freed.
//! Everything above that one cell was never streaming — it was residency wearing
//! a cache's clothes, and paying a cache's overhead.
//!
//! # The floating cell costs nothing when nothing streams
//!
//! It exists only when the budget cannot hold the trunk. A zone that fits every
//! layer plans no cell at all, so a model that fits pays exactly its own bytes
//! and the mechanism is inert — no idle reservation, no transfer, no fence.
//! When the boundary later concedes ground to KV, the first layer given up frees
//! its own space and the cell is carved from it.
//!
//! # Order, and why the frontier end is the least protected
//!
//! Layers are laid down from `span_end` **downward** in [`protection_order`], so
//! the most protected sits against the dense block where retraction can never
//! reach, and the frontier the boundary eats into holds the layer the eviction
//! order gives up first. Retraction is therefore truncation: pop entries off the
//! end of the walk, and both the eviction order and the addresses agree without
//! anything being computed or moved.

use super::descriptor::LayerImage;
use super::order::protection_order;

/// Where one layer's bytes are, and how many of them there are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPlacement {
    /// Device address of the layer image's offset 0.
    pub base: u64,
    /// The image's `total` — its own size, not a cell's.
    pub bytes: usize,
}

/// The zone's layout for one byte budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZonePlan {
    /// `layer → its fixed home`, or `None` when the layer streams.
    pub homes: Vec<Option<LayerPlacement>>,
    /// The one cell streamed layers pass through, absent when none do.
    pub floating: Option<LayerPlacement>,
    /// Streaming layers in [`super::eviction_order`], which is the order they
    /// were given up in and the order they are taken back in.
    pub missing: Vec<usize>,
    /// Lowest address the plan occupies. Everything below is the KV side's.
    pub floor: u64,
}

impl ZonePlan {
    /// Layers with a home.
    pub fn resident(&self) -> usize {
        self.homes.iter().filter(|h| h.is_some()).count()
    }

    /// Bytes between [`Self::floor`] and the zone's top.
    ///
    /// **Not the budget it was planned against.** Dense packing leaves a
    /// remainder too small for another layer, and the carve hands that back to
    /// KV rather than fencing it off — under equal cells the same remainder was
    /// simply lost inside the last cell.
    pub fn used_bytes(&self, end: u64) -> usize {
        (end - self.floor) as usize
    }

    /// Whether every layer is resident, so nothing ever crosses PCIe again.
    pub fn is_whole(&self) -> bool {
        self.floating.is_none()
    }
}

/// Errors a budget can have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZoneError {
    /// The budget cannot hold the pinned head and a floating cell — the floor
    /// below which the model cannot run at all, not merely run slowly.
    TooSmall { need: usize, have: usize },
}

impl std::fmt::Display for ZoneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooSmall { need, have } => write!(
                f,
                "layer zone: {} MiB will not hold the pinned head and one streaming cell ({} MiB)",
                have >> 20,
                need >> 20
            ),
        }
    }
}

impl std::error::Error for ZoneError {}

/// Lay `images` out downward from `end` within `budget` bytes.
///
/// `pinned` leading layers have no record in the cold tier, so they cannot
/// stream and are always given a home; a budget that will not hold them is an
/// error rather than a degraded plan.
pub fn plan_zone(
    images: &[LayerImage],
    pinned: usize,
    end: u64,
    budget: usize,
) -> Result<ZonePlan, ZoneError> {
    let n = images.len();
    let pinned = pinned.min(n);
    let order = protection_order(n, pinned);
    let whole: usize = images.iter().map(|i| i.total).sum();

    // **Sized for the largest layer that could ever stream, not for the largest
    // that streams now.** The missing set grows as the boundary concedes ground,
    // so a cell sized to today's set would have to be reallocated — and moving
    // it moves the floor, which is the one address the KV side has already been
    // told about. One number, fixed at load.
    let cell = images
        .iter()
        .enumerate()
        .filter(|(l, _)| *l >= pinned)
        .map(|(_, i)| i.total)
        .max()
        .unwrap_or(0);

    let pinned_bytes: usize = images.iter().take(pinned).map(|i| i.total).sum();
    if whole > budget && pinned_bytes + cell > budget {
        return Err(ZoneError::TooSmall {
            need: pinned_bytes + cell,
            have: budget,
        });
    }
    if whole <= budget {
        return Ok(lay_out(images, &order, n, None, end));
    }

    // How many layers of the protection order fit beside the cell. Walked
    // rather than divided, because the layers are not the same size — which is
    // the whole reason this tier exists.
    let room = budget - cell;
    let mut used = 0usize;
    let mut resident = 0usize;
    for &l in &order {
        let want = images[l].total;
        if used + want > room {
            break;
        }
        used += want;
        resident += 1;
    }
    Ok(lay_out(images, &order, resident, Some(cell), end))
}

/// Place the first `resident` of `order` downward from `end`, then the cell.
fn lay_out(
    images: &[LayerImage],
    order: &[usize],
    resident: usize,
    cell: Option<usize>,
    end: u64,
) -> ZonePlan {
    let mut homes = vec![None; images.len()];
    let mut at = end;
    for &l in order.iter().take(resident) {
        let bytes = images[l].total;
        at -= bytes as u64;
        homes[l] = Some(LayerPlacement { base: at, bytes });
    }
    let floating = cell.map(|bytes| {
        at -= bytes as u64;
        LayerPlacement { base: at, bytes }
    });
    // The eviction order restricted to what has no home — which, because the
    // resident set is a prefix of the protection order, is exactly a prefix of
    // the eviction order reversed back into it.
    let missing: Vec<usize> = order
        .iter()
        .skip(resident)
        .rev()
        .copied()
        .filter(|&l| homes[l].is_none())
        .collect();
    ZonePlan {
        homes,
        floating,
        missing,
        floor: at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::layer_stream::descriptor::{
        layer_image, FfnForm, LayerTensor, MixKind, Projection, PROJECTION_ALIGN,
    };
    use candle::quantized::GgmlDType;

    /// A DeltaNet image whose `total` is `units × PROJECTION_ALIGN`.
    fn image(units: usize) -> LayerImage {
        let roles = [
            LayerTensor::Wqkv,
            LayerTensor::Wz,
            LayerTensor::WOut,
            LayerTensor::FfnGateUp,
            LayerTensor::FfnDown,
        ];
        // Four projections of one unit and one carrying the remainder, so the
        // image total is exactly what the caller asked for.
        let rest = units - (roles.len() - 1);
        let p: Vec<Projection> = roles
            .iter()
            .enumerate()
            .map(|(i, &role)| Projection {
                role,
                shape: [32, 128],
                dtype: GgmlDType::Q4_KO,
                payload: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
                extent: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
            })
            .collect();
        layer_image(MixKind::DeltaNet, FfnForm::Fused, &p).unwrap()
    }

    const END: u64 = 1 << 40;

    /// Sizes chosen so the layers are *not* equal — the case equal cells get
    /// wrong. Layer 0 is the largest, as it is on the real checkpoint.
    fn model() -> Vec<LayerImage> {
        (0..16)
            .map(|l| image(if l == 0 { 20 } else { 10 + l % 3 }))
            .collect()
    }

    #[test]
    fn a_budget_that_holds_everything_plans_no_cell() {
        let m = model();
        let whole: usize = m.iter().map(|i| i.total).sum();
        let p = plan_zone(&m, 2, END, whole).unwrap();
        assert!(p.is_whole(), "nothing should stream");
        assert!(p.floating.is_none());
        assert!(p.missing.is_empty());
        assert_eq!(p.resident(), m.len());
        // And it claims exactly the model's bytes — no cell, no rounding to a
        // cell multiple, so the remainder of the budget stays with KV.
        assert_eq!(p.used_bytes(END), whole);
    }

    /// Bytes below which the model cannot run: the pinned head plus one cell.
    fn floor_bytes(m: &[LayerImage], pinned: usize) -> usize {
        m.iter().take(pinned).map(|i| i.total).sum::<usize>() + stream_cell(m, pinned)
    }

    /// The cell the planner will choose — the largest *streamable* layer, which
    /// is not the largest layer when the biggest one is pinned (it is, on the
    /// real checkpoint: layer 0 is both).
    fn stream_cell(m: &[LayerImage], pinned: usize) -> usize {
        m.iter().skip(pinned).map(|i| i.total).max().unwrap()
    }

    /// The saving over equal cells, stated as the property rather than a number:
    /// dense packing never holds fewer layers in the same bytes, and holds
    /// strictly more as soon as the budget is past the first few cells.
    ///
    /// The tie at the smallest budgets is real and worth recording — this
    /// model's largest layer is also its first, and a pinned layer has to be
    /// resident whatever it costs, so at three cells both schemes fit the same
    /// three layers. The gain arrives when there is room for ordinary layers,
    /// which is every budget the zone actually runs at.
    #[test]
    fn dense_packing_holds_at_least_as_many_layers_as_equal_cells() {
        let m = model();
        let cell = m.iter().map(|i| i.total).max().unwrap();
        for budget_cells in 3..=8 {
            let budget = budget_cells * cell;
            let dense = plan_zone(&m, 2, END, budget).unwrap().resident();
            assert!(
                dense >= budget_cells,
                "{budget_cells} cells held {dense} layers dense"
            );
            // Except where dense packing has already taken the whole model and
            // there is nothing left to be better at.
            if budget_cells >= 4 && dense < m.len() {
                assert!(
                    dense > budget_cells,
                    "{budget_cells} cells held {dense} layers dense"
                );
            }
        }
    }

    /// Placements descend from `end`, never overlap, and the floor is below all
    /// of them. An overlap here is two layers computing with each other's bytes.
    #[test]
    fn placements_are_disjoint_and_descend_from_the_top() {
        let m = model();
        let cell = m.iter().map(|i| i.total).max().unwrap();
        let p = plan_zone(&m, 2, END, 5 * cell).unwrap();
        let mut spans: Vec<(u64, u64)> = p
            .homes
            .iter()
            .flatten()
            .chain(p.floating.iter())
            .map(|h| (h.base, h.base + h.bytes as u64))
            .collect();
        spans.sort_unstable();
        assert_eq!(
            spans.first().unwrap().0,
            p.floor,
            "the cell sits at the floor"
        );
        assert_eq!(
            spans.last().unwrap().1,
            END,
            "the top layer touches the end"
        );
        for w in spans.windows(2) {
            assert_eq!(w[0].1, w[1].0, "a gap or an overlap at {w:?}");
        }
    }

    /// The pinned head always has a home, and takes the addresses furthest from
    /// the frontier.
    #[test]
    fn the_pinned_head_is_resident_and_furthest_from_the_frontier() {
        let m = model();
        let cell = m.iter().map(|i| i.total).max().unwrap();
        let p = plan_zone(&m, 2, END, 4 * cell).unwrap();
        let h0 = p.homes[0].expect("layer 0 pinned");
        let h1 = p.homes[1].expect("layer 1 pinned");
        assert_eq!(h0.base + h0.bytes as u64, END);
        assert_eq!(h1.base + h1.bytes as u64, h0.base);
        assert!(!p.missing.contains(&0) && !p.missing.contains(&1));
    }

    /// **Growth takes layers back in the order they were given up.** The
    /// resident set is a prefix of the protection order at every budget, so a
    /// larger budget is a superset — nothing already resident is displaced, and
    /// the missing set stays a prefix of the eviction order and so stays spread.
    #[test]
    fn a_larger_budget_is_a_superset_and_takes_back_the_last_given_up() {
        let m = model();
        let whole: usize = m.iter().map(|i| i.total).sum();
        let lo = floor_bytes(&m, 2);
        let mut prev: Option<ZonePlan> = None;
        for step in 0..=16 {
            let p = plan_zone(&m, 2, END, lo + step * (whole - lo) / 16).unwrap();
            if let Some(q) = &prev {
                assert!(p.resident() >= q.resident(), "residency went backwards");
                for (l, h) in q.homes.iter().enumerate() {
                    if h.is_some() {
                        assert!(p.homes[l].is_some(), "layer {l} was displaced by growth");
                    }
                }
                // What growth took back is the tail of the previous missing set,
                // i.e. the layers most recently given up.
                let took: Vec<usize> = q
                    .missing
                    .iter()
                    .copied()
                    .filter(|&l| p.homes[l].is_some())
                    .collect();
                let tail: Vec<usize> = q.missing[q.missing.len() - took.len()..].to_vec();
                assert_eq!(took, tail, "growth took back out of order");
            }
            prev = Some(p);
        }
    }

    /// A budget under the floor is an error, not a plan that cannot run. The
    /// failure it replaces was *"L2 is absent and no slot can hold it"* on the
    /// next forward, with nothing to retry.
    #[test]
    fn a_budget_below_the_pinned_head_and_one_cell_is_refused() {
        let m = model();
        let need = floor_bytes(&m, 2);
        assert!(matches!(
            plan_zone(&m, 2, END, need - 1),
            Err(ZoneError::TooSmall { .. })
        ));
        let p = plan_zone(&m, 2, END, need).unwrap();
        assert_eq!(p.resident(), 2, "at the floor, only the pinned head fits");
    }

    /// The cell holds the *largest* streamable layer, not the largest currently
    /// streaming one — so conceding more ground never has to resize it.
    #[test]
    fn the_cell_is_sized_for_any_layer_that_could_ever_stream() {
        let m = model();
        let biggest = stream_cell(&m, 2);
        let lo = floor_bytes(&m, 2);
        let whole: usize = m.iter().map(|i| i.total).sum();
        for step in 0..16 {
            let p = plan_zone(&m, 2, END, lo + step * (whole - lo) / 16).unwrap();
            if let Some(f) = p.floating {
                assert_eq!(f.bytes, biggest, "budget {step} cells");
                for &l in &p.missing {
                    assert!(m[l].total <= f.bytes, "layer {l} will not fit the cell");
                }
            }
        }
    }

    /// `missing` is in eviction order: the layer given up first comes first, so
    /// a shrink appends and a growth pops.
    #[test]
    fn missing_is_in_eviction_order() {
        let m = model();
        let cell = m.iter().map(|i| i.total).max().unwrap();
        let p = plan_zone(&m, 2, END, 4 * cell).unwrap();
        let evict = crate::models::layer_stream::eviction_order(m.len(), 2);
        let want: Vec<usize> = evict.into_iter().take(p.missing.len()).collect();
        assert_eq!(p.missing, want);
    }
}
