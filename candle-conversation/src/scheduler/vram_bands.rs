//! The whole card, cut by what holds each byte, once per wave — what the
//! performance page stacks over time.
//!
//! Every band is a measurement, and together they are the card: driver free,
//! what was on the card before the span was reserved, the CUDA pool outside the
//! span, the span's own partition (`candle_nn::kv_cache::SpanPartition`), and
//! whatever is left, named `unaccounted` and signed. A chart that only shows
//! some of them cannot say where the rest went; this one adds up to the card's
//! total at every wave.

use candle_nn::kv_cache::SpanPartition;

/// One wave's card, in MiB.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VramBands {
    /// The card's size, as the driver reports it.
    pub total: f64,
    /// Free, as the driver reports it.
    pub free: f64,
    /// On the card before the span was reserved: the CUDA context, module
    /// images, and any other process.
    pub context: f64,
    /// The CUDA async pool, outside the span: allocations no ground took.
    pub pool: f64,
    /// The model's dense weights.
    pub dense: f64,
    /// Routed experts resident in the weight zone.
    pub experts: f64,
    /// The weight zone's slots holding no expert.
    pub expert_free: f64,
    /// KV arenas.
    pub kv: f64,
    /// Span tenants: recurrent state, the provenance gallery, index caches.
    pub state: f64,
    /// The widest forward's activations since the last wave.
    pub transient: f64,
    /// Fixed at reservation: the held ground, the persistence staging, and the
    /// span's alignment tails.
    pub reserved: f64,
    /// KV regions nobody holds.
    pub kv_free: f64,
    /// What the card holds that no band names: `total − free − every band`.
    /// Signed, and meant to sit near zero.
    pub unaccounted: f64,
}

const MIB: f64 = 1024.0 * 1024.0;

/// The bands, from the driver's `total` and `free`, the device bytes in use
/// before the span (`context`, when it was measured), the CUDA pool's reserved
/// bytes, the span's partition, and the resident experts' bytes — all bytes.
pub fn vram_bands(
    total: u64,
    free: u64,
    context: Option<u64>,
    pool: u64,
    span: Option<SpanPartition>,
    resident_experts: u64,
) -> VramBands {
    let span = span.unwrap_or_default();
    let experts = (resident_experts as usize).min(span.weight_zone);
    let reserved = span.held_ground + span.persist_staging + span.dense_padding + span.slack;
    let in_use = total.saturating_sub(free) as i64;
    let context = context.unwrap_or(0);
    let unaccounted = in_use - context as i64 - pool as i64 - span.sum() as i64;
    let mib = |b: usize| b as f64 / MIB;
    VramBands {
        total: total as f64 / MIB,
        free: free as f64 / MIB,
        context: context as f64 / MIB,
        pool: pool as f64 / MIB,
        dense: mib(span.dense),
        experts: mib(experts),
        expert_free: mib(span.weight_zone - experts),
        kv: mib(span.kv),
        state: mib(span.tenants),
        transient: mib(span.transient),
        reserved: mib(reserved),
        kv_free: mib(span.regions_free),
        unaccounted: unaccounted as f64 / MIB,
    }
}

impl VramBands {
    /// Every band but `total` — the card's total, by construction.
    pub fn sum(&self) -> f64 {
        self.free
            + self.context
            + self.pool
            + self.dense
            + self.experts
            + self.expert_free
            + self.kv
            + self.state
            + self.transient
            + self.reserved
            + self.kv_free
            + self.unaccounted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: u64 = 1024 * 1024;

    fn span() -> SpanPartition {
        SpanPartition {
            held_ground: 64 * M as usize,
            persist_staging: 448 * M as usize,
            dense: 9_000 * M as usize,
            dense_padding: 8 * M as usize,
            kv: 12_000 * M as usize,
            tenants: 2_000 * M as usize,
            transient: 900 * M as usize,
            regions_free: 4_000 * M as usize,
            slack: 4 * M as usize,
            weight_zone: 40_000 * M as usize,
        }
    }

    /// **The bands are the card.** A 72 GiB card with 1 GiB free, 1.2 GiB of
    /// context, 300 MiB of pool and a span whose weight zone holds 38,000 MiB
    /// of experts: every byte lands in one band, and the residual is the 2,752
    /// MiB of the card the span and the rest do not explain.
    #[test]
    fn the_bands_add_up_to_the_card() {
        let total = 73_728 * M;
        let free = 1_024 * M;
        let b = vram_bands(
            total,
            free,
            Some(1_228 * M),
            300 * M,
            Some(span()),
            38_000 * M,
        );
        assert_eq!(
            b,
            VramBands {
                total: 73_728.0,
                free: 1_024.0,
                context: 1_228.0,
                pool: 300.0,
                dense: 9_000.0,
                experts: 38_000.0,
                expert_free: 2_000.0,
                kv: 12_000.0,
                state: 2_000.0,
                transient: 900.0,
                reserved: 524.0,
                kv_free: 4_000.0,
                unaccounted: 2_752.0,
            }
        );
        assert_eq!(b.sum(), b.total);
    }

    /// No span yet — the model still loading — is every byte in use read as
    /// context, pool or unaccounted, and the sum still holds.
    #[test]
    fn a_card_with_no_span_still_closes() {
        let b = vram_bands(16_384 * M, 15_000 * M, None, 100 * M, None, 0);
        assert_eq!((b.kv, b.dense, b.unaccounted), (0.0, 0.0, 1_284.0));
        assert_eq!(b.sum(), b.total);
    }

    /// More experts reported than the zone holds is the zone, full.
    #[test]
    fn experts_never_read_wider_than_their_zone() {
        let b = vram_bands(73_728 * M, 0, Some(0), 0, Some(span()), 50_000 * M);
        assert_eq!((b.experts, b.expert_free), (40_000.0, 0.0));
    }
}
