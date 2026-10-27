//! How many rows one prefill forward carries, by the card's VRAM and its
//! measured host→device link.
//!
//! A forward's transient tier grows with its rows — about 0.9 MiB a row on
//! Qwen3.8-Flash-Next — and where the model's experts already want the whole
//! card, that tier is bought from the expert zone: the zone concedes the
//! ground for the cohort's sweep and grows back empty after it, and decode
//! steps on the misses until the experts are copied back over the link. Width
//! pays prefill by spreading the forward's fixed cost and its copy of the
//! experts that are not resident, `(experts − resident) / link`, over more
//! rows. So the two dimensions pull against each other:
//!
//! * **The VRAM**: how large the expert zone is beside the other tenants, and
//!   so how much of it a tier is. A small zone loses a large share of itself
//!   to any tier; a card that holds the whole model with room to spare
//!   concedes nothing, and its width is limited by prefill compute alone.
//! * **The link**: on a slow one the copy a forward amortises is seconds, so
//!   width earns prefill a lot, and the dip it leaves refills slowly and costs
//!   every miss inside it more. On a fast one the copy is small, prefill is
//!   compute-bound by a few thousand rows, and every row past that is ground
//!   decode loses for nothing.
//!
//! Measured on Qwen3.8-Flash-Next (BlackWorld, RTX PRO 5000 72 GB, PCIe 5.0,
//! 2026-10-27): its prefill reaches its compute ceiling by ~3.7k rows — 6,715
//! tok/s at ~3.7k, 6,639 at ~7.3k, 6,191 at ~14.5k — while an 8,192-row
//! cohort bought an ~8 GB tier from the experts every window, the zone sat at
//! ~17 GB of ~60 GB and the decode hit rate at 52–68%. The 16 GB row is
//! measured too: 4,096 there starved admission (`SchedulerConfig::default`).
//! The other cells are modelled from those two and the gate's prefill curve,
//! and are what a width sweep on each machine refines.
//!
//! [`MAX_PREFILL_TOKENS`](candle_transformers::models::batched_inference) is
//! the model side's per-forward ceiling (8,192): the cards that hold the whole
//! model with room to spare sit at it.

/// VRAM bands' lower bounds, ascending: what a card reports, a little under
/// its nameplate (a 3090 reads ~23.7 GiB, a 64 GB card under 64 GiB).
const VRAM_BANDS: [u64; 7] = [
    0,               // 16 GB class: the expert zone at its floor
    22 * (1 << 30),  // 24 GB class
    28 * (1 << 30),  // 32 GB class (RTX 5090)
    40 * (1 << 30),  // 48 GB class
    60 * (1 << 30),  // 64–80 GB class (RTX PRO 5000 72 GB)
    88 * (1 << 30),  // 96 GB class: Flash-Next resident with ~20 GB spare
    120 * (1 << 30), // 141 GB and up
];

/// Link bands' lower bounds, bytes a second, ascending: PCIe 3.0 ×16 and
/// 4.0 ×8 (~12 GB/s), 4.0 ×16 (~25), 5.0 ×16 (~50), 6.0 ×16 (~100), 7.0 ×16
/// (~200), NVLink-C2C (450–900).
const LINK_BANDS: [f64; 6] = [0.0, 18e9, 37e9, 75e9, 150e9, 300e9];

/// Rows per prefill forward: `WIDTH[vram band][link band]`.
const WIDTH: [[usize; 6]; 7] = [
    [2048, 2048, 2048, 2048, 2048, 2048],
    [3072, 4096, 4096, 4096, 4096, 3072],
    [4096, 4096, 4096, 4096, 3072, 3072],
    [4096, 4096, 3072, 3072, 2048, 2048],
    [6144, 4096, 3072, 2048, 2048, 2048],
    [8192, 8192, 8192, 8192, 8192, 8192],
    [8192, 8192, 8192, 8192, 8192, 8192],
];

/// The band `value` falls in: the last lower bound it reaches.
fn band<T: PartialOrd>(bounds: &[T], value: T) -> usize {
    bounds.iter().rposition(|b| value >= *b).unwrap_or(0)
}

/// Rows one prefill forward carries on a card reporting `total_vram_bytes`
/// whose pinned host→device copies run at `link_bytes_per_s`. A card with no
/// reading is read as the smallest card, and a link with none as the slowest
/// band.
pub(crate) fn prefill_pass_tokens(total_vram_bytes: u64, link_bytes_per_s: f64) -> usize {
    let link = if link_bytes_per_s.is_finite() && link_bytes_per_s > 0.0 {
        link_bytes_per_s
    } else {
        0.0
    };
    WIDTH[band(&VRAM_BANDS, total_vram_bytes)][band(&LINK_BANDS, link)]
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The machines this runs on, at what each reports and measures.
    #[test]
    fn each_machine_takes_its_cell() {
        assert_eq!(
            prefill_pass_tokens(16 * GIB, 12e9),
            2048,
            "RTX 4090 Mobile, ×8"
        );
        assert_eq!(
            prefill_pass_tokens(23 * GIB + 700 * (1 << 20), 12e9),
            3072,
            "RTX 3090 behind PCIe 3.0"
        );
        assert_eq!(
            prefill_pass_tokens(72 * GIB, 52e9),
            3072,
            "RTX PRO 5000, PCIe 5.0"
        );
    }

    /// Every cell, against the raw table.
    #[test]
    fn every_band_reads_its_cell() {
        let vram = [16, 24, 32, 48, 72, 96, 141].map(|g| g * GIB);
        let link = [12e9, 25e9, 52e9, 100e9, 200e9, 900e9];
        let expected = [
            [2048, 2048, 2048, 2048, 2048, 2048],
            [3072, 4096, 4096, 4096, 4096, 3072],
            [4096, 4096, 4096, 4096, 3072, 3072],
            [4096, 4096, 3072, 3072, 2048, 2048],
            [6144, 4096, 3072, 2048, 2048, 2048],
            [8192, 8192, 8192, 8192, 8192, 8192],
            [8192, 8192, 8192, 8192, 8192, 8192],
        ];
        for (v, row) in vram.iter().zip(expected) {
            for (l, want) in link.iter().zip(row) {
                assert_eq!(
                    prefill_pass_tokens(*v, *l),
                    want,
                    "{} GiB at {l:e} B/s",
                    v / GIB
                );
            }
        }
    }

    /// A band starts at its bound: one under is the band below.
    #[test]
    fn a_band_starts_at_its_bound() {
        assert_eq!(prefill_pass_tokens(60 * GIB, 12e9), 6144, "64–80 GB class");
        assert_eq!(prefill_pass_tokens(60 * GIB - 1, 12e9), 4096, "48 GB class");
        assert_eq!(prefill_pass_tokens(72 * GIB, 75e9), 2048, "PCIe 6.0 band");
        assert_eq!(
            prefill_pass_tokens(72 * GIB, 75e9 - 1.0),
            3072,
            "PCIe 5.0 band"
        );
    }

    /// A card with no VRAM reading is the smallest card; a link with no
    /// reading is the slowest band.
    #[test]
    fn no_reading_falls_to_the_first_band() {
        assert_eq!(prefill_pass_tokens(0, 52e9), 2048);
        assert_eq!(prefill_pass_tokens(72 * GIB, f64::NAN), 6144);
        assert_eq!(prefill_pass_tokens(72 * GIB, 0.0), 6144);
    }

    /// Never past the model side's per-forward ceiling.
    #[test]
    fn never_past_the_models_ceiling() {
        assert!(WIDTH.iter().flatten().all(|&w| w <= 8192));
    }
}
