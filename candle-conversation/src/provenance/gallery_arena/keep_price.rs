//! The weight side of the gallery's keep trade: what one more byte of expert
//! ground would save, per second, measured from the expert cache's tallies.
//!
//! Ground the gallery gives up goes back to the span, where the weight zone
//! grows into it. Each slot it gains holds one more expert, and the misses
//! that slot spares are measured, not assumed: the pipeline thread counts the
//! routed experts a zone a window of slots larger would have served under
//! LRU (`candle-transformers` `expert_lre::stack_distance`), so over the
//! window that count is the misses one more slot spares. Each spared miss
//! saves the stall a miss costs, which the pipeline thread prices as it
//! serves them — a copy over the link for a pinned miss, the measured pack
//! read for a cold one — so the stall over the misses is the stall per miss.
//! Per byte of the slot:
//!
//! ```text
//! marginal hits/s ÷ window × stall µs per miss ÷ slot bytes     µs/s per byte
//! ```
//!
//! **Why not the misses spread evenly over the experts not resident.** That
//! was the price: one more slot keeps one more of the experts not resident,
//! so `1 / (experts − resident slots)` of the misses hit, each sparing a link
//! copy. Routing is skewed, so the expert the next slot would keep is routed
//! far more than the average one past the zone, and a cold miss waits on the
//! drive, not the link. On WhiteWorld (2026-11-22) it priced a slot at a
//! 23,551th of 263 misses a second, set the keep time to 3,778 s, and the
//! 7.2 GiB gallery zend's preload left on the 24 GB card held its ground
//! beside a 900-slot expert zone hitting 12–45%.
//!
//! The rates are averaged over [`RATE_TAU_SECS`]: the tallies are read every
//! scheduler step, and a step's count is too few routings to price by.

use std::time::Instant;

/// Time constant of the rates' averages, in seconds.
pub const RATE_TAU_SECS: f64 = 10.0;

/// The expert cache's tallies as read: monotonic within an interval, and
/// restarted at the cache's stats reset.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WeightTallies {
    /// Routed experts that missed.
    pub misses: usize,
    /// Routed experts a zone `window` slots larger would have served.
    pub marginal_hits: usize,
    /// The misses' stall, µs.
    pub stall_us: f64,
}

/// The rates behind the weight side's price, averaged as they are read.
#[derive(Debug, Clone, Default)]
pub struct KeepPrice {
    seen: Option<(Instant, WeightTallies)>,
    /// Misses a second.
    pub misses_per_s: f64,
    /// Marginal hits a second, over the whole window.
    pub marginal_per_s: f64,
    /// Stall µs a second.
    pub stall_us_per_s: f64,
}

impl KeepPrice {
    /// The tallies read at `now` join the averages. A tally below the last one
    /// read was restarted, and is all new.
    pub fn observe(&mut self, now: Instant, t: WeightTallies) {
        if let Some((then, seen)) = self.seen {
            let secs = now.saturating_duration_since(then).as_secs_f64();
            if secs > 0.0 {
                let w = 1.0 - (-secs / RATE_TAU_SECS).exp();
                let new = |now: usize, was: usize| if now >= was { now - was } else { now };
                let new_stall = if t.stall_us >= seen.stall_us {
                    t.stall_us - seen.stall_us
                } else {
                    t.stall_us
                };
                let step = |avg: &mut f64, n: f64| *avg += w * (n / secs - *avg);
                step(&mut self.misses_per_s, new(t.misses, seen.misses) as f64);
                step(
                    &mut self.marginal_per_s,
                    new(t.marginal_hits, seen.marginal_hits) as f64,
                );
                step(&mut self.stall_us_per_s, new_stall);
            } else {
                // Read twice at one instant: the earlier reading stands.
                return;
            }
        }
        self.seen = Some((now, t));
    }

    /// The stall one miss costs, µs; zero before a miss is measured.
    pub fn stall_us_per_miss(&self) -> f64 {
        if self.misses_per_s > 0.0 {
            self.stall_us_per_s / self.misses_per_s
        } else {
            0.0
        }
    }

    /// What one more byte of weight ground saves, µs a second, with the
    /// marginal hits counted over `window` slots of `slot_bytes` each. Zero
    /// when nothing would be spared — no marginal hit, no miss, or no slot to
    /// price — which keeps everything the gallery holds.
    pub fn weight_us_per_byte_sec(&self, window: usize, slot_bytes: usize) -> f64 {
        if window == 0 || slot_bytes == 0 {
            return 0.0;
        }
        let per_slot = self.marginal_per_s / window as f64;
        per_slot * self.stall_us_per_miss() / slot_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const SLOT: usize = 3 << 20;

    /// Prices `secs` seconds of steady tallies, read once a second.
    fn steady(misses: usize, marginal: usize, stall_us: f64, secs: u64) -> KeepPrice {
        let mut p = KeepPrice::default();
        let t0 = Instant::now();
        for s in 0..=secs {
            p.observe(
                t0 + Duration::from_secs(s),
                WeightTallies {
                    misses: misses * s as usize,
                    marginal_hits: marginal * s as usize,
                    stall_us: stall_us * s as f64,
                },
            );
        }
        p
    }

    /// **No miss, no price** — the card that holds its experts (BlackWorld):
    /// nothing marginal, nothing stalled, and the gallery keeps all it holds.
    #[test]
    fn a_zone_that_holds_its_experts_puts_no_price_on_ground() {
        let p = steady(0, 0, 0.0, 120);
        assert_eq!(p.weight_us_per_byte_sec(64, SLOT), 0.0);
        assert_eq!(p.stall_us_per_miss(), 0.0);
        let fresh = KeepPrice::default();
        assert_eq!(fresh.weight_us_per_byte_sec(64, SLOT), 0.0, "nothing read");
    }

    /// Misses that one more slot would not spare put no price on ground
    /// either: a streaming working set far past the zone.
    #[test]
    fn misses_no_slot_would_spare_put_no_price_on_ground() {
        let p = steady(300, 0, 300.0 * 2_000.0, 120);
        assert!(p.stall_us_per_miss() > 1_900.0);
        assert_eq!(p.weight_us_per_byte_sec(112, SLOT), 0.0);
    }

    /// **The price is the marginal hits per slot times the stall per miss,
    /// per byte.** Steady for two minutes the averages have converged to
    /// within e^-12: 300 misses/s costing 2 ms each, 56 marginal hits/s over
    /// a 112-slot window — half a miss a second per slot, 1 ms/s, per 3 MiB.
    #[test]
    fn the_price_is_the_marginal_stall_per_byte() {
        let p = steady(300, 56, 300.0 * 2_000.0, 120);
        let want = 0.5 * 2_000.0 / SLOT as f64;
        let got = p.weight_us_per_byte_sec(112, SLOT);
        assert!((got / want - 1.0).abs() < 1e-4, "{got} vs {want}");
    }

    /// The cache's stats reset restarts its tallies: a reading below the last
    /// is all new, not a negative rate.
    #[test]
    fn a_restarted_tally_is_counted_as_new() {
        let mut p = KeepPrice::default();
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);
        let tallies = |m, h, st| WeightTallies {
            misses: m,
            marginal_hits: h,
            stall_us: st,
        };
        p.observe(at(0), tallies(1_000, 100, 1e6));
        p.observe(at(1), tallies(10, 1, 1e4));
        assert!(p.misses_per_s > 0.0 && p.marginal_per_s > 0.0 && p.stall_us_per_s > 0.0);
        let w = 1.0 - (-1.0 / RATE_TAU_SECS).exp();
        assert!((p.misses_per_s - 10.0 * w).abs() < 1e-9);
        // A second read at the same instant changes nothing.
        let before = p.clone();
        p.observe(at(1), tallies(500, 50, 5e5));
        assert_eq!(p.misses_per_s, before.misses_per_s);
    }

    /// No window or no slot size is no price, never a division by zero.
    #[test]
    fn an_unmeasured_geometry_is_no_price() {
        let p = steady(300, 56, 6e5, 30);
        assert_eq!(p.weight_us_per_byte_sec(0, SLOT), 0.0);
        assert_eq!(p.weight_us_per_byte_sec(112, 0), 0.0);
    }
}
