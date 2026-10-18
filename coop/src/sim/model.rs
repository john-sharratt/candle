//! A member as the simulator runs it: what a decode step and a prefill forward
//! cost it, and how that grows as the conversations it holds take ground from
//! its expert zone.
//!
//! The step costs are BlackWorld's fitted curve (`docs/cooperative_admission.md`
//! §1); a member not yet measured carries its gate figures scaled onto the same
//! shape, and says so ([`MemberModel::measured`]).

/// A member's costs.
#[derive(Debug, Clone, PartialEq)]
pub struct MemberModel {
    pub name: &'static str,
    /// Whether these come from its own wave log, or are assumed.
    pub measured: bool,
    /// `CAPACITY`'s power, which today's `Best` divides.
    pub power: u32,
    /// A decode step: `decode_base_ms + decode_row_ms · rows`, at full
    /// residency.
    pub decode_base_ms: f64,
    pub decode_row_ms: f64,
    /// The most rows one decode step carries.
    pub max_rows: usize,
    /// A prefill forward: `prefill_base_ms + prefill_token_ms · tokens +
    /// prefill_mkv_ms · MKV`, at full residency.
    pub prefill_base_ms: f64,
    pub prefill_token_ms: f64,
    pub prefill_mkv_ms: f64,
    /// The most prompt tokens one prefill forward carries.
    pub prefill_width: usize,
    /// A decode step's slowdown while prefill waits beside it.
    pub beside_prefill: f64,
    /// The expert zone with nothing held, in GB, and what each conversation
    /// the member holds takes from it.
    pub zone_gb: f64,
    pub conversation_gb: f64,
}

/// The experts a decode batch keeps coming back to, in GB: BlackWorld's
/// heavy-decode windows read 0.93–0.98 with a 35 GB zone.
const DECODE_WORKING_GB: f64 = 37.0;

/// The same for prefill of large tool results: its heavy-read windows read
/// 0.38 with 22.5 GB.
const PREFILL_WORKING_GB: f64 = 59.0;

/// The best hit rate seen.
const TOP_HIT: f64 = 0.97;

/// How much slower every forward runs per unit of hit rate lost below the
/// top: 1.7× at 0.38 (BlackWorld's heavy-read windows).
const SLOWDOWN_PER_HIT: f64 = 1.2;

impl MemberModel {
    /// BlackWorld, RTX PRO 5000 72 GB: measured, builds 3436–3442.
    pub fn black_world() -> MemberModel {
        MemberModel {
            name: "BlackWorld",
            measured: true,
            power: 734,
            decode_base_ms: 45.0,
            decode_row_ms: 1.4,
            max_rows: 64,
            prefill_base_ms: 13.7,
            prefill_token_ms: 0.218,
            prefill_mkv_ms: 12.7,
            prefill_width: 2048,
            beside_prefill: 1.25,
            zone_gb: 58.0,
            conversation_gb: 0.6,
        }
    }

    /// WhiteWorld, RTX 3090 24 GB: assumed from its gate (×16 decode a third
    /// of BlackWorld's, prefill half) until its wave log is fitted.
    pub fn white_world() -> MemberModel {
        MemberModel {
            name: "WhiteWorld",
            measured: false,
            power: 156,
            decode_base_ms: 45.0,
            decode_row_ms: 4.5,
            max_rows: 32,
            prefill_base_ms: 13.7,
            prefill_token_ms: 0.45,
            prefill_mkv_ms: 25.0,
            prefill_width: 1024,
            beside_prefill: 1.25,
            zone_gb: 9.0,
            conversation_gb: 0.6,
        }
    }

    /// The expert zone left while holding `held` conversations, in GB.
    pub fn zone(&self, held: usize) -> f64 {
        (self.zone_gb - self.conversation_gb * held as f64).max(0.0)
    }

    /// A decode step's expert hit rate while holding `held`.
    pub fn decode_hit(&self, held: usize) -> f64 {
        (self.zone(held) / DECODE_WORKING_GB).min(TOP_HIT)
    }

    /// A prefill forward's expert hit rate while holding `held`.
    pub fn prefill_hit(&self, held: usize) -> f64 {
        (self.zone(held) / PREFILL_WORKING_GB).min(TOP_HIT)
    }

    /// How much slower a forward runs at `hit` than at the top.
    pub fn slowdown(hit: f64) -> f64 {
        1.0 + SLOWDOWN_PER_HIT * (TOP_HIT - hit).max(0.0)
    }

    /// One decode step of `rows`, holding `held`, with or without prefill
    /// waiting beside it.
    pub fn decode_ms(&self, rows: usize, held: usize, prefill_waiting: bool) -> f64 {
        let step = self.decode_base_ms + self.decode_row_ms * rows as f64;
        let beside = if prefill_waiting {
            self.beside_prefill
        } else {
            1.0
        };
        step * beside * Self::slowdown(self.decode_hit(held))
    }

    /// One prefill forward of `tokens` over `kv` attended positions, holding
    /// `held`.
    pub fn prefill_ms(&self, tokens: usize, kv: f64, held: usize) -> f64 {
        let fwd = self.prefill_base_ms
            + self.prefill_token_ms * tokens as f64
            + self.prefill_mkv_ms * kv / 1e6;
        fwd * Self::slowdown(self.prefill_hit(held))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BlackWorld's measured points come back: one row alone ~46 ms, 23 rows
    /// ~77 ms, both at the top hit rate its 58 GB zone reads.
    #[test]
    fn black_world_reproduces_its_measurements() {
        let b = MemberModel::black_world();
        assert_eq!(b.decode_hit(0), 0.97);
        assert!((b.decode_ms(1, 0, false) - 46.4).abs() < 1e-9);
        assert!((b.decode_ms(23, 0, false) - 77.2).abs() < 1e-9);
        assert!((b.decode_ms(23, 0, true) - 96.5).abs() < 1e-9);
    }

    /// The measured hit rates come back from the zone: 24 chats leave 43.6 GB
    /// and decode stays at the top; 59 GB of read context leaves 22.5 GB
    /// under a read prefill's 59 GB working set, 0.38.
    #[test]
    fn the_zone_sets_the_hit_rate() {
        let b = MemberModel::black_world();
        assert_eq!(b.decode_hit(24), 0.97);
        let read = MemberModel {
            conversation_gb: 35.5 / 25.0,
            ..b.clone()
        };
        assert!((read.prefill_hit(25) - 0.381).abs() < 1e-3);
    }

    /// Holding more conversations takes zone and slows every forward: with
    /// none left, forwards run 2.16× slower.
    #[test]
    fn holding_conversations_slows_forwards() {
        let b = MemberModel::black_world();
        assert!(b.decode_hit(80) < b.decode_hit(0));
        assert_eq!(b.decode_hit(100), 0.0);
        assert!((MemberModel::slowdown(0.0) - 2.164).abs() < 1e-9);
        assert!(b.prefill_ms(2048, 0.0, 100) > b.prefill_ms(2048, 0.0, 0));
    }
}
