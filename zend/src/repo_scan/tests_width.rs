//! Tests for the width arithmetic: what the weight side may concede, what a
//! conversation costs, and which bounds hold.
//!
//! The estimate must price the KV the scan adds and not the corpus it inherits,
//! and the numerator of the width must price the same bytes as its denominator.
//! The weight zone's floor is not room the scan may plan against, so a width
//! takes only what the zone can concede above its floor. The floor charges
//! reserved arena, not the live half. Whatever the inputs, the width stays between
//! one and the pool's ceiling: it is never zero, since a pool that may not open a
//! conversation can never free the VRAM it waits for.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// **The concession is measured against the ZONE, never base + zone.**
    ///
    /// `floor_bytes` is a fraction of the elastic expert zone's achievable
    /// residency, and dense base weights live outside the span it is measured
    /// in — no concession can release them. Counting them overstated the ground
    /// a scan could take by `base_bytes / 2`, and `scan_width`'s own tests
    /// cannot see it because the unit is chosen at the call site.
    #[test]
    fn what_the_weight_side_holds_is_the_zone_not_the_dense_base() {
        use candle_conversation::memory_report::WeightSection;
        let section = |base: Option<u64>, floor: Option<u64>| WeightSection {
            base_bytes: base,
            resident_expert_bytes: Some(40 << 30),
            floor_bytes: floor,
        };
        assert_eq!(
            weights_now(&section(Some(20 << 30), Some(30 << 30))),
            40 << 30,
            "the dense base is not the zone's and is not conceded",
        );
        assert_eq!(
            weights_now(&section(None, Some(30 << 30))),
            40 << 30,
            "a stack reporting no base reads the same — which is why this was latent",
        );
        assert_eq!(
            weights_now(&section(Some(20 << 30), None)),
            u64::MAX,
            "no floor to defend ⇒ unbounded, so the caller's `min` cannot clamp \
             the width to nothing",
        );
    }

    /// Measured report from a `repo_map` pass on the 16 GiB card, 21 directories
    /// in: capacity 13.17 GiB, scratch margin 1 GiB, `pool_used` 9.67 GiB, KV
    /// arenas 2.31 GiB.
    pub(in super::super) const CAPACITY: u64 = 14_143_193_088;

    pub(in super::super) const SCRATCH: u64 = 1_073_741_824;

    pub(in super::super) const POOL_USED: u64 = 10_380_898_464;

    pub(in super::super) const KV: u64 = 2_483_027_968;

    /// Arenas standing before the first directory opened — tool sections, the
    /// base builder's prefill, and the calibration exemplars.
    pub(in super::super) const PRE_SCAN: u64 = 2_000_000_000;

    /// The estimate must price the KV a scan ADDS. Charged the whole process's
    /// arenas instead, the per-conversation estimate pinned to its ceiling — and
    /// the error compounds, since a narrower pool divides the same fixed corpus
    /// into a still-larger estimate.
    #[test]
    fn the_estimate_prices_the_scans_own_kv_not_the_inherited_corpus() {
        assert!(per_conversation_kv(KV, PRE_SCAN, 2) < per_conversation_kv(KV, 0, 2));
        assert_eq!(per_conversation_kv(KV, 0, 2), SCAN_CONV_KV_MAX);
    }

    /// ...and the numerator must price it the same way. `fixed` is
    /// `pool_used - kv`, so subtracting only `fixed` returns every KV byte as
    /// free room — including the inherited corpus the denominator deliberately
    /// excludes. Growing that corpus leaves the same room for new work, so the
    /// pool must get narrower; unfixed, it stayed exactly as wide.
    #[test]
    fn the_inherited_corpus_is_not_free_room() {
        const EXTRA: u64 = 2 * 1024 * 1024 * 1024;
        let base = scan_width(CAPACITY, SCRATCH, POOL_USED, KV, PRE_SCAN, 2, 0, u64::MAX);
        let with_corpus = scan_width(
            CAPACITY,
            SCRATCH,
            POOL_USED + EXTRA,
            KV + EXTRA,
            PRE_SCAN + EXTRA,
            2,
            0,
            u64::MAX,
        );
        assert!(with_corpus < base, "{with_corpus} vs {base}");
    }

    /// Holding back a FRACTION of capacity double-charges the expert cache:
    /// `fixed` subtracts it explicitly, then the fraction holds back a share of
    /// capacity that is mostly the same bytes again. On the measured report a
    /// 0.70 fraction left 1.87 GiB for scan KV against the governor's own
    /// ~5 GiB KV floor — under half the room, and the pool ran that much
    /// narrower for it.
    #[test]
    fn the_margin_held_back_is_the_governors_not_a_fraction_of_capacity() {
        let per_conv = per_conversation_kv(KV, PRE_SCAN, 2);
        let held_by_fraction = CAPACITY - ((CAPACITY as f64) * 0.70) as u64;
        assert!(held_by_fraction > SCRATCH * 3, "{held_by_fraction}");
        let by_fraction = (((CAPACITY as f64) * 0.70) as u64)
            .saturating_sub(POOL_USED - KV + PRE_SCAN)
            / per_conv;
        let by_governor =
            scan_width(CAPACITY, SCRATCH, POOL_USED, KV, PRE_SCAN, 2, 0, u64::MAX) as u64;
        assert_eq!(by_fraction, 0);
        assert_eq!(by_governor, 6);
    }

    /// **The weight zone's floor is not room the scan may plan against.**
    ///
    /// K/V and the model's weights share one span, so every region the pool
    /// claims is ground the weight side loses. A width computed from capacity
    /// alone therefore plans to evict the model — and did: 96 directory
    /// conversations grew 48 GB of K/V, drove the weight zone from 53.5 GB to
    /// 10.9 GB against a 28.8 GB floor, and left admission refusing every
    /// prefill from then on, because the floor check compares residency to the
    /// floor and refusing a prefill gives no K/V back.
    #[test]
    fn the_scan_may_take_only_what_the_weight_zone_can_concede() {
        let unbounded = scan_width(CAPACITY, SCRATCH, POOL_USED, KV, PRE_SCAN, 2, 0, u64::MAX);
        // A zone holding 9 GiB over an 8 GiB floor may concede 1 GiB, which is
        // less than the capacity arithmetic would have allowed.
        let defended = scan_width(
            CAPACITY,
            SCRATCH,
            POOL_USED,
            KV,
            PRE_SCAN,
            2,
            8 * 1024 * 1024 * 1024,
            9 * 1024 * 1024 * 1024,
        );
        assert!(defended < unbounded, "{defended} vs {unbounded}");
        // A zone already at its floor concedes nothing, and the pool still does
        // not reach zero: one conversation has to be able to run, or the ingest
        // cannot make progress at all.
        assert_eq!(
            scan_width(
                CAPACITY,
                SCRATCH,
                POOL_USED,
                KV,
                PRE_SCAN,
                2,
                8 * 1024 * 1024 * 1024,
                8 * 1024 * 1024 * 1024,
            ),
            1,
        );
        // Ground the weights already hold is not charged to the scan: a zone
        // far above its floor is bounded by capacity, exactly as before.
        assert_eq!(
            scan_width(
                CAPACITY,
                SCRATCH,
                POOL_USED,
                KV,
                PRE_SCAN,
                2,
                8 * 1024 * 1024 * 1024,
                u64::MAX,
            ),
            unbounded,
        );
    }

    /// The floor charges a conversation for RESERVED arena, not for the live
    /// half — and that is the width the card actually sustains.
    ///
    /// Pricing only the live ~240 MiB (on the theory that slab waste belongs to
    /// a format rather than to a conversation) opens 19 directories on this
    /// report and walks straight into the wall: measured, 15 directories lost in
    /// one millisecond, a device OOM, 18 MiB free, and a 1415 MiB gap between
    /// reserved and used arena that is precisely the overhead the halving
    /// assumed away. Ten is the width that ran clean, and faster.
    #[test]
    fn the_floor_charges_reserved_arena_not_just_the_live_half() {
        assert_eq!(SCAN_CONV_KV_MIN, 480 * 1024 * 1024);
        let for_kv = CAPACITY - SCRATCH - (POOL_USED - KV) - PRE_SCAN;
        assert_eq!(for_kv / SCAN_CONV_KV_MIN, 6);
        // Halving the floor nearly doubles the width — the same argument that
        // opened 19 directories on this card and walked into the wall.
        assert_eq!(for_kv / (256 * 1024 * 1024), 11);
    }

    /// Growing the inherited corpus must not inflate the per-conversation
    /// estimate: the anchor moves with it, so the same in-flight KV is priced
    /// the same however large the corpus underneath it grows.
    #[test]
    fn a_larger_inherited_corpus_does_not_inflate_the_estimate() {
        const EXTRA: u64 = 4 * 1024 * 1024 * 1024;
        assert_eq!(
            per_conversation_kv(KV, PRE_SCAN, 2),
            per_conversation_kv(KV + EXTRA, PRE_SCAN + EXTRA, 2),
        );
    }

    /// A pool holding nothing open has added nothing, so the estimate is the
    /// measured floor rather than a division over a corpus it never created.
    #[test]
    fn an_idle_pool_estimates_at_the_floor() {
        assert_eq!(per_conversation_kv(9_000_000_000, 0, 0), SCAN_CONV_KV_MIN);
        assert_eq!(
            per_conversation_kv(9_000_000_000, 9_000_000_000, 4),
            SCAN_CONV_KV_MIN,
        );
        // Demotion can hand back more than the scan added; the delta floors.
        assert_eq!(
            per_conversation_kv(1_000, 9_000_000_000, 4),
            SCAN_CONV_KV_MIN
        );
    }

    /// The estimate stays inside its measured clamps at both ends: a cold start
    /// cannot admit unboundedly, and one atypical directory cannot stall the
    /// scan by pricing every later one off the card.
    #[test]
    fn the_per_conversation_estimate_stays_within_its_measured_clamps() {
        assert_eq!(
            per_conversation_kv(4 * SCAN_CONV_KV_MAX, 0, 1),
            SCAN_CONV_KV_MAX,
        );
        assert_eq!(
            per_conversation_kv(SCAN_CONV_KV_MIN / 4, 0, 1),
            SCAN_CONV_KV_MIN,
        );
        assert_eq!(
            per_conversation_kv(2 * SCAN_CONV_KV_MIN, 0, 2),
            SCAN_CONV_KV_MIN,
        );
    }

    /// The bound is a width, so it never reaches zero — a pool that may not open
    /// a conversation can never free the VRAM it is waiting on — and never
    /// exceeds the thread count the pool actually spawns.
    #[test]
    fn the_width_stays_between_one_and_the_pool_ceiling() {
        assert_eq!(
            scan_width(CAPACITY, SCRATCH, 14_000_000_000, 0, 0, 0, 0, u64::MAX),
            1,
        );
        // A margin wider than the card leaves nothing, and still not zero.
        assert_eq!(
            scan_width(CAPACITY, 2 * CAPACITY, 0, 0, 0, 0, 0, u64::MAX),
            1,
        );
        assert_eq!(
            scan_width(1024 * 1024 * 1024 * 1024, SCRATCH, 0, 0, 0, 0, 0, u64::MAX),
            REPO_MAP_PARALLELISM,
        );
    }
}
