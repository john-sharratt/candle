//! Choosing an expert weight format from the card the model is landing on.
//!
//! The engine's conversion emits `Q4_KO` because that is what the released
//! W4A16 source *is* — `group_size=128 symmetric` is the same 4-bit grid, so
//! `qwen4exp::convert` is a bit-exact re-encoding and not a quantization at all.
//! That is the right answer on a card with room for it, and the wrong one on a
//! card without: a 512-expert MoE's resident zone is the model's whole
//! footprint, so the format decides whether it fits.
//!
//! This is the policy that decides, keyed on total VRAM.
//!
//! # Two things this ladder is not, both worth saying out loud
//!
//! **Below `Q4_KO` it is a REQUANTIZATION, not a repack.** The `Q4_KO` rung
//! re-encodes an existing 4-bit grid and loses nothing; the rung under it
//! re-quantizes 4-bit weights to 3 bits and loses something real. A caller
//! dropping a rung is buying residency with accuracy, and should know it.
//!
//! **The floor is `Q3_KO`, by choice.** `Q2_KO` exists and is narrower, but
//! [`GgmlDType::to_ko`]'s own note prices it: its four levels "floor at
//! `rel_l2 ≈ 0.325`, a quality loss rather than a repack". The expert cache
//! streams what does not fit (VRAM → pinned RAM → NVMe), so a narrower format
//! buys speed on a small card, never feasibility — and that is not worth a
//! third of the weights' signal. Every card under the `Q4_KO` rung takes
//! `Q3_KO`; [`below_narrowest_rung`] names the cards too small for any rung.

use candle::quantized::GgmlDType;
use candle::{Device, Result};

/// The rungs, in descending VRAM order: a card with **at least** `gib` gibibytes
/// takes the format above it, so the search is "first rung whose floor this card
/// clears".
///
/// Read as: under 86 GiB take `Q4_KO`, under 64 take `Q3_KO`. Above 86 GiB
/// nothing is forced and the source's own width stands.
const LADDER: &[(u64, GgmlDType)] = &[(86, GgmlDType::Q4_KO), (64, GgmlDType::Q3_KO)];

/// The narrowest format the ladder hands out — its last rung, and every card
/// under 64 GiB. Not the narrowest KO format that exists; see the module docs
/// for why the ladder stops above `Q2_KO`.
pub const FLOOR_KO: GgmlDType = GgmlDType::Q3_KO;

/// The expert format for a card of `vram_gib`, or `None` to leave the source's
/// own width alone.
///
/// `None` above the top rung is deliberate: a card that can hold the model at
/// the source's width should not pay a requantization for nothing.
pub fn expert_format(vram_gib: u64) -> Option<GgmlDType> {
    let mut pick = None;
    for &(floor, fmt) in LADDER {
        if vram_gib < floor {
            pick = Some(fmt);
        }
    }
    pick
}

/// The **drafter's** expert format: the trunk's, always.
///
/// A draft head looks like the place to give width up — it is one block against
/// forty-eight, and its only job is to propose tokens the target then checks, so
/// a worse proposal costs a rejected slot rather than a wrong answer. That
/// argument is why this used to return the narrowest KO format, and it is wrong here
/// for a reason particular to how the expert zone is laid out.
///
/// **Slots are uniformly sized to the widest layer** — `slot_bytes_for` takes a
/// `max` over every layer's geometry — so a layer's format is not a local
/// decision:
///
/// * A **wider** drafter inflates every slot in the grid (49 layers × 512
///   experts), roughly doubling the zone's footprint for one block's benefit and
///   halving how many experts stay resident. The cache runs 98.5% warm; that
///   would end it.
/// * A **narrower** drafter under-fills its slots and frees nothing. It saves
///   PCIe bytes per draft step and no residency at all.
///
/// So the only width that costs nothing is the trunk's, and the acceptance a
/// narrower head would give up is throughput the drafter exists to buy. If a
/// profile ever shows draft-step expert DMA as the wall, narrowing is the lever
/// — and it is a bandwidth trade, not a residency one.
pub fn drafter_format(vram_gib: u64) -> Option<GgmlDType> {
    expert_format(vram_gib)
}

/// Whether this card is under every rung the ladder serves.
///
/// The ladder was specified with a `< 12 GiB → Q1_KO` rung and there is no
/// `Q1_KO`; a caller at that size is asking for a width the ladder does not
/// hand out. [`expert_format`] still answers [`FLOOR_KO`], so this is the
/// predicate a loader uses to refuse rather than to quietly under-serve.
pub fn below_narrowest_rung(vram_gib: u64) -> bool {
    vram_gib < 12
}

/// This device's total VRAM in gibibytes, for feeding the functions above.
pub fn device_vram_gib(device: &Device) -> Result<u64> {
    let (_free, total) = device.mem_get_info()?;
    Ok(nominal_gib(total as u64))
}

/// A card's nominal size in GiB from the total the driver reports.
///
/// **Rounded, not floored.** The driver reports a little under the nominal
/// size — the RTX 4090 Mobile reads 16,375.5 MiB of its 16 GiB — so flooring
/// puts every card one GiB below its own rung: a 64 GiB card would read 63 and
/// fall to the `Q3_KO` rung the ladder gives 32 GiB cards.
pub fn nominal_gib(total_bytes: u64) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    (total_bytes + GIB / 2) / GIB
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    /// Reported totals from the fleet land on their nominal sizes.
    #[test]
    fn reported_totals_round_to_the_card() {
        // RTX 4090 Mobile: 16,375.5 MiB reported.
        assert_eq!(nominal_gib(16_375 * MIB + MIB / 2), 16);
        // A card reading just under 24 GiB, and one just under 32.
        assert_eq!(nominal_gib(24_100 * MIB), 24);
        assert_eq!(nominal_gib(32_600 * MIB), 32);
        // Exactly on a size, and just past the half-way point below it.
        assert_eq!(nominal_gib(16 * 1024 * MIB), 16);
        assert_eq!(nominal_gib(15 * 1024 * MIB + 512 * MIB), 16);
        assert_eq!(nominal_gib(15 * 1024 * MIB + 511 * MIB), 15);
    }

    /// The fleet, by the machine each figure belongs to — the sizing table in
    /// CLAUDE.md names its machine per row for exactly this reason.
    #[test]
    fn the_ladder_places_every_card_in_the_fleet() {
        // RTX PRO 5000 Blackwell, 72 GiB — under 86, so Q4_KO, which is also
        // what the bit-exact W4A16 re-encoding already produces. The big card
        // pays nothing for the ladder existing.
        assert_eq!(expert_format(72), Some(GgmlDType::Q4_KO));
        // 2× RTX 5090, 32 GiB each — under 64, so Q3_KO.
        assert_eq!(expert_format(32), Some(GgmlDType::Q3_KO));
        // RTX 3090, 24 GiB — likewise Q3_KO.
        assert_eq!(expert_format(24), Some(GgmlDType::Q3_KO));
        // RTX 4090 Mobile, 16 GiB — likewise Q3_KO, the floor.
        assert_eq!(expert_format(16), Some(GgmlDType::Q3_KO));
        // Above every rung: leave the source alone rather than requantize for
        // no reason.
        assert_eq!(expert_format(96), None);
        assert_eq!(expert_format(86), None);
    }

    /// The boundaries are `<`, not `<=`, on every rung — a card sitting exactly
    /// on a floor takes the wider format.
    #[test]
    fn a_card_exactly_on_a_floor_takes_the_wider_rung() {
        assert_eq!(expert_format(85), Some(GgmlDType::Q4_KO));
        assert_eq!(expert_format(64), Some(GgmlDType::Q4_KO));
        assert_eq!(expert_format(63), Some(GgmlDType::Q3_KO));
    }

    /// The drafter takes the trunk's width on every card, because slots are
    /// sized to the widest layer and so a head's format is a whole-grid
    /// decision — see [`drafter_format`].
    #[test]
    fn the_drafter_matches_the_trunk_on_every_card() {
        for gib in [16, 24, 32, 72, 85, 96] {
            assert_eq!(
                drafter_format(gib),
                expert_format(gib),
                "the drafter is one layer of the same grid, so it takes the grid's width"
            );
        }
    }

    /// The ladder stops at `Q3_KO`: every small card takes it, and the sub-12
    /// GiB case is a refusal rather than a narrower format.
    #[test]
    fn nothing_is_handed_out_below_the_floor() {
        assert!(below_narrowest_rung(8));
        assert!(!below_narrowest_rung(12));
        assert_eq!(FLOOR_KO, GgmlDType::Q3_KO);
        for gib in [8, 12, 16, 24, 31, 32, 63] {
            assert_eq!(expert_format(gib), Some(FLOOR_KO), "{gib} GiB");
        }
    }
}
