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
/// else, draws 37 ms of `quantize`, 5 ms of `housekeeping` and 1,195 ms of
/// `blocked` — the remainder nothing named.
#[test]
fn the_gap_quantize_is_its_own_band_and_out_of_blocked() {
    let mut w = WaveStats::new();
    w.add_quantize(37_400);
    w.add_housekeeping(5_000);
    w.push_phase_window(1_237);
    let mut bands = bands_of(1_237);
    bands.sort();
    assert_eq!(
        bands,
        vec![("blocked", 1_195), ("housekeeping", 5), ("quantize", 37)]
    );
}

/// **Compaction and the telemetry flush are named apart from housekeeping.**
/// A 1,241 ms window that timed 12 ms of compaction, 9 ms of the last window's
/// flush and 5 ms of other housekeeping draws each as its own band, and only
/// the 1,215 ms nothing timed as `blocked`.
#[test]
fn compaction_and_the_flush_are_named_apart_from_housekeeping() {
    let mut w = WaveStats::new();
    w.add_compaction(12_300);
    w.add_telemetry(9_100);
    w.add_housekeeping(5_000);
    w.push_phase_window(1_241);
    let mut bands = bands_of(1_241);
    bands.sort();
    assert_eq!(
        bands,
        vec![
            ("blocked", 1_215),
            ("compaction", 12),
            ("housekeeping", 5),
            ("telemetry", 9)
        ]
    );
}

/// A window with no gap work draws no `quantize` band at all.
#[test]
fn a_window_without_gap_work_draws_no_quantize_band() {
    let mut w = WaveStats::new();
    w.add_housekeeping(5_000);
    w.push_phase_window(1_239);
    assert!(bands_of(1_239).iter().all(|(kind, _)| *kind != "quantize"));
}
