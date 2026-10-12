//! The engine's columns of a telemetry sample, read off the shared scheduler.
//!
//! The scheduler already records what it does — per-wave phase measurements
//! and per-wave summaries (`candle_conversation::phase_ring`) — so the
//! telemetry sampler reads the engine there rather than waiting to be told. One
//! scheduler serves every tenant, so these columns are the engine's, not any
//! one tenant's.

use candle_conversation::phase_ring::{snapshot, snapshot_waves};

use super::sample::Engine;

/// One phase measurement, reduced to what a rate needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseTokens {
    pub kind: &'static str,
    /// Seconds before now.
    pub age_s: f64,
    pub tokens: u64,
}

/// The newest wave's memory split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveMemory {
    pub kv_mib: u64,
    pub weights_mib: u64,
}

/// The engine reading for the last `window_s` seconds: decode and prefill
/// tokens per second over the window, and the newest wave's KV and weights.
/// `None` when the scheduler has recorded no wave at all — no engine, which is
/// not the same as an idle one.
pub fn engine_reading(
    phases: &[PhaseTokens],
    newest_wave: Option<WaveMemory>,
    window_s: f64,
) -> Option<Engine> {
    let wave = newest_wave?;
    let rate = |kinds: &[&str]| {
        let tokens: u64 = phases
            .iter()
            .filter(|p| p.age_s <= window_s && kinds.contains(&p.kind))
            .map(|p| p.tokens)
            .sum();
        tokens as f64 / window_s
    };
    Some(Engine {
        decode_tps: Some(rate(&["decode"])),
        prefill_tps: Some(rate(&["prefill"])),
        kv_mib: Some(wave.kv_mib as f64),
        weights_mib: Some(wave.weights_mib as f64),
        ..Engine::default()
    })
}

/// [`engine_reading`] over the scheduler's live rings.
pub fn engine_now(window_s: f64) -> Option<Engine> {
    let phases: Vec<PhaseTokens> = snapshot()
        .into_iter()
        .map(|p| PhaseTokens {
            kind: p.kind,
            age_s: p.age_s,
            tokens: p.tokens,
        })
        .collect();
    let newest = snapshot_waves()
        .into_iter()
        .min_by(|a, b| a.age_s.total_cmp(&b.age_s))
        .map(|w| WaveMemory {
            kv_mib: w.sample.quant_mib + w.sample.float_mib,
            weights_mib: w.sample.weights_mib,
        });
    engine_reading(&phases, newest, window_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(kind: &'static str, age_s: f64, tokens: u64) -> PhaseTokens {
        PhaseTokens {
            kind,
            age_s,
            tokens,
        }
    }

    #[test]
    fn no_wave_is_no_engine() {
        assert!(engine_reading(&[p("decode", 0.5, 10)], None, 2.0).is_none());
    }

    /// Decode counts decode phases; prefill counts prefill phases, section
    /// ingest included; anything older than the window is out of it.
    #[test]
    fn rates_count_the_window_and_memory_is_the_newest_wave() {
        let phases = [
            p("decode", 0.5, 30),
            p("decode", 1.5, 50),
            p("decode", 3.0, 1_000),
            p("prefill", 1.0, 4_000),
            p("prefill", 1.9, 2_000),
            p("sealing", 1.0, 99),
        ];
        let e = engine_reading(
            &phases,
            Some(WaveMemory {
                kv_mib: 512,
                weights_mib: 20_000,
            }),
            2.0,
        )
        .unwrap();
        assert_eq!(e.decode_tps, Some(40.0));
        assert_eq!(e.prefill_tps, Some(3_000.0));
        assert_eq!(e.kv_mib, Some(512.0));
        assert_eq!(e.weights_mib, Some(20_000.0));
        assert_eq!(e.npcs_active, None, "a tenant's column is the tenant's");
    }

    /// An idle engine reads zero — measured, and nothing ran.
    #[test]
    fn an_idle_engine_reads_zero_not_absent() {
        let e = engine_reading(
            &[],
            Some(WaveMemory {
                kv_mib: 0,
                weights_mib: 1,
            }),
            2.0,
        )
        .unwrap();
        assert_eq!(e.decode_tps, Some(0.0));
    }
}
