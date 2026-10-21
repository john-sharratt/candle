//! Qwen3.8-Flash-Next's speculation, swept on each card it runs on.
//!
//! Its NextN head is a full routed block of the same architecture, folded into
//! the artifact as `blk.48`, and its draft ceiling at each wave width is fitted
//! from these sweeps and blended by the card's VRAM ([`super::draft_blend`]).
//!
//! **Measured** by `quantized_qwen38_moe::tests::spec_sweep::spec_sweep_wNN`
//! (2026-10-21): one process per width, 256 generated tokens a session,
//! StoryRewrite, every point passing the rewrite verdict. Each width's rates are
//! decode tokens a second at budgets 0, 1, 2, …, the sweep stopping at the first
//! budget slower than the one before it. Budget 0 is the width's *closing*
//! baseline, measured after every budget: the opening one ran while the expert
//! zone was still warming and read 5–22% low, which would credit speculation
//! with the warm-up.
//!
//! **What it replaced.** The ladder was `[(8, 4), (16, 2)]`: budget 4 measured
//! to width 8 on the 72 GB card, then the lineage's inherited 2 to width 16, and
//! nothing wider. Both cards still pay well past it: at width 15 budget 2 is
//! 1,280 t/s on the 72 GB card against 1,639 at 4, and at width 32 — where the
//! ladder drafted nothing — budget 4 is 2,125 against 922 plain. The 24 GB card
//! settles at 3 from width 6, at ~1.5× plain through width 19.
//!
//! **Past width 32 is unmeasured** on either card; the blend carries each
//! card's widest ceiling on (`DraftTable::budget`). Neither card's gain had
//! begun to fall by 32.
//!
//! **Ceilings, not depths.** The rewrite is the text the head predicts best,
//! so the sweep finds the most depth that ever pays at a width; each sequence
//! drafts below it at its own acceptance (`super::draft_depth`). Free
//! continuation accepts ~2.1 a step (`docs/performance.md` §3.6): on the 72 GB
//! card alone (2026-10-06) the essay drafted every step at depth 1, 2, 3 ran
//! 81, 97, 84 t/s, and under the depth rule at ceiling 4 the rewrite ran 236
//! t/s at 4.90 accepted a step and the essay 88 at 2.14.
//!
//! **The head's own ceiling is 4** (`QWEN38_MAX_DRAFT` in the sweep): a
//! one-session ceiling of 5 lost on Strata's benchmark (2026-10-07), 3.0–3.45
//! accepted a step against ~3.07 at 4 for ~2 ms more a step. No ceiling may
//! pass 7: a block is the drafts and the row they follow, and it must fit each
//! DeltaNet layer's 8-row timeline (`docs/deltanet_timeline.md`).

use super::draft_blend::{CardSweep, SweepWidth};

/// What one drafted token costs relative to a step's fixed cost — the `c / T0`
/// the depth rule prices a depth with (`super::draft_depth`). Measured on the
/// essay's ~4.3K context: `2.3 / 18.4 ≈ 0.125` (the rewrite's ~1.3K gives
/// 0.085), and swept as (rewrite, essay) t/s: 0.085 → (313, 83), 0.125 →
/// (324, 88), 0.15 → (313, 87), 0.2 → (290, 96).
pub const TOKEN_COST: f32 = 0.125;

const fn w(width: usize, rates: &'static [f64]) -> SweepWidth {
    SweepWidth { width, rates }
}

/// The GeForce RTX 3090, 24 GB, behind PCIe 3.0 (WhiteWorld): widths 1–20,
/// then 24, 28 and 32 — its ceiling had settled at 2 from 16, and the widths
/// between take the one before them.
pub const CARD_24GB: CardSweep = CardSweep {
    vram_mib: 24_576,
    widths: &[
        w(1, &[44.7, 70.0, 78.7, 83.1, 87.7]),
        w(2, &[81.8, 128.3, 142.1, 153.6, 152.4]),
        w(3, &[105.9, 157.7, 181.8, 201.0, 205.0]),
        w(4, &[127.4, 190.1, 212.2, 233.0, 234.9]),
        w(5, &[148.3, 211.1, 250.5, 271.1, 276.9]),
        w(6, &[165.9, 225.6, 261.2, 281.9, 280.6]),
        w(7, &[170.4, 240.4, 276.8, 293.0, 289.3]),
        w(8, &[180.8, 262.3, 300.1, 318.2, 319.2]),
        w(9, &[188.5, 272.0, 305.8, 335.2, 344.1]),
        w(10, &[201.2, 297.8, 337.3, 354.2, 349.8]),
        w(11, &[210.4, 302.9, 350.3, 361.4, 356.0]),
        w(12, &[236.5, 316.1, 346.2, 362.6, 371.9]),
        w(13, &[238.4, 320.8, 357.9, 365.6, 361.3]),
        w(14, &[237.1, 324.4, 357.0, 369.0, 351.7]),
        w(15, &[235.2, 323.5, 349.2, 355.0, 336.1]),
        w(16, &[244.2, 329.0, 352.7, 353.6, 341.5]),
        w(17, &[239.6, 330.8, 350.4, 341.4]),
        w(18, &[251.6, 358.4, 384.6, 390.0, 376.0]),
        w(19, &[251.0, 349.7, 370.8, 370.2]),
        w(20, &[260.9, 352.9, 371.8, 363.9]),
        w(24, &[290.7, 381.6, 416.1, 405.8]),
        w(28, &[306.5, 415.5, 437.1, 432.2]),
        w(32, &[327.8, 444.1, 460.1, 452.5]),
    ],
};

/// The RTX PRO 5000 Blackwell, 72 GB (BlackWorld), widths 1–32.
pub const CARD_72GB: CardSweep = CardSweep {
    vram_mib: 73_415,
    widths: &[
        w(1, &[88.8, 133.6, 183.8, 228.4, 254.5]),
        w(2, &[168.7, 255.3, 363.4, 429.7, 450.5]),
        w(3, &[233.5, 357.6, 476.0, 490.2, 605.3]),
        w(4, &[301.5, 450.2, 598.2, 611.5, 690.1]),
        w(5, &[351.2, 511.4, 693.2, 719.9, 878.4]),
        w(6, &[402.5, 582.4, 749.6, 865.1, 974.1]),
        w(7, &[451.2, 644.8, 835.1, 919.7, 1034.5]),
        w(8, &[499.0, 661.8, 861.3, 1039.2, 1199.5]),
        w(9, &[507.3, 771.3, 1005.1, 1162.9, 1323.2]),
        w(10, &[533.2, 806.3, 1047.6, 1103.4, 1330.3]),
        w(11, &[563.9, 852.9, 1090.6, 1179.5, 1463.7]),
        w(12, &[580.0, 822.2, 1117.9, 1282.3, 1392.6]),
        w(13, &[603.3, 874.5, 1189.6, 1411.5, 1416.2]),
        w(14, &[626.5, 851.3, 1151.8, 1360.0, 1486.9]),
        w(15, &[656.8, 948.2, 1280.3, 1365.7, 1638.7]),
        w(16, &[678.8, 955.5, 1322.9, 1542.1, 1724.2]),
        w(17, &[702.3, 982.5, 1352.9, 1436.3, 1749.3]),
        w(18, &[722.7, 1017.8, 1402.7, 1641.5, 1819.3]),
        w(19, &[730.8, 1073.6, 1390.0, 1521.0, 1786.6]),
        w(20, &[733.7, 1108.5, 1440.5, 1638.0, 1653.9]),
        w(21, &[758.7, 1136.2, 1470.3, 1596.7, 1745.6]),
        w(22, &[775.4, 1130.6, 1503.4, 1651.1, 1804.4]),
        w(23, &[780.9, 1139.2, 1543.4, 1681.0, 1833.7]),
        w(24, &[794.9, 1156.4, 1540.1, 1861.7, 1853.7]),
        w(25, &[817.4, 1178.8, 1618.7, 1688.0, 1925.3]),
        w(26, &[826.0, 1208.9, 1653.2, 1736.6, 1930.4]),
        w(27, &[848.9, 1231.6, 1660.6, 1764.9, 1958.1]),
        w(28, &[855.7, 1312.1, 1605.9, 1742.0, 1992.0]),
        w(29, &[865.1, 1322.2, 1630.9, 1776.6, 2018.4]),
        w(30, &[882.0, 1344.2, 1560.7, 1700.0, 1983.8]),
        w(31, &[903.0, 1365.4, 1687.0, 1852.0, 2093.5]),
        w(32, &[921.5, 1414.0, 1707.6, 1904.9, 2125.1]),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::delta_net::TIMELINE_ROWS;
    use crate::models::draft_blend::blend;

    /// **Each swept card reads its own fitted row.** The 72 GB card's best is 4
    /// at nearly every width, its few 3s within a per cent of 4; the 24 GB
    /// card's is 4 to width 5, 3 to 15, then 2 — noise pooled away, never
    /// rising with width.
    #[test]
    fn each_card_reads_its_own_row() {
        assert_eq!(
            blend(&CARD_24GB, &CARD_72GB, CARD_72GB.vram_mib),
            vec![4; 32]
        );
        let mut low = vec![4; 5];
        low.extend([3; 10]);
        low.extend([2; 17]);
        assert_eq!(blend(&CARD_24GB, &CARD_72GB, CARD_24GB.vram_mib), low);
    }

    /// A card between the two takes ceilings between theirs, never above the
    /// larger nor below the smaller; and no ceiling, at any VRAM, passes what
    /// the DeltaNet timeline can verify.
    #[test]
    fn a_card_between_is_between_and_inside_the_timeline() {
        let (lo, hi) = (
            blend(&CARD_24GB, &CARD_72GB, CARD_24GB.vram_mib),
            blend(&CARD_24GB, &CARD_72GB, CARD_72GB.vram_mib),
        );
        for vram in [16_384, 32_768, 49_152, 98_304] {
            let row = blend(&CARD_24GB, &CARD_72GB, vram);
            for (w, &b) in row.iter().enumerate() {
                assert!(
                    b >= lo[w].min(hi[w]) && b <= lo[w].max(hi[w]),
                    "{vram} MiB w{}",
                    w + 1
                );
                assert!(b < TIMELINE_ROWS, "{vram} MiB drafts {b} at w{}", w + 1);
            }
        }
    }
}
