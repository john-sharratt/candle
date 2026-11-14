//! A wave window's phase split: what the loop timed between its forwards is
//! named, and only what nothing timed is left as `blocked`.

use super::phase_ring::snapshot;
use super::WaveStats;

/// The window's bands as `(kind, dur_ms)`, found by a `window_ms` no other
/// test pushes — the ring is process-global.
fn bands_of(window_ms: u32) -> Vec<(&'static str, u32)> {
    snapshot()
        .into_iter()
        .filter(|p| p.window_ms == window_ms)
        .map(|p| (p.kind, p.dur_ms))
        .collect()
}

/// **The gap quantize is its own band, carved out of `blocked`.** A 1,237 ms
/// window that timed 37 ms of gap work and 5 ms of housekeeping, and nothing
/// else, draws 37 ms of `kv_gap`, 5 ms of `housekeeping` and 1,195 ms of
/// `blocked` — the remainder nothing named.
#[test]
fn the_gap_quantize_is_its_own_band_and_out_of_blocked() {
    let mut w = WaveStats::new();
    w.add_kv_gap(37_400);
    w.add_housekeeping(5_000);
    w.push_phase_window(1_237);
    let mut bands = bands_of(1_237);
    bands.sort();
    assert_eq!(
        bands,
        vec![("blocked", 1_195), ("housekeeping", 5), ("kv_gap", 37)]
    );
}

/// A window with no gap work draws no `kv_gap` band at all.
#[test]
fn a_window_without_gap_work_draws_no_gap_band() {
    let mut w = WaveStats::new();
    w.add_housekeeping(5_000);
    w.push_phase_window(1_239);
    assert!(bands_of(1_239).iter().all(|(kind, _)| *kind != "kv_gap"));
}
