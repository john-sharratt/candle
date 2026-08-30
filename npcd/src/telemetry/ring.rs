//! The history the performance page draws.
//!
//! # Why the history lives here and not in the browser
//!
//! The first version of this page had no ring: `/v1/telemetry` returned one
//! instantaneous reading and the console accumulated its own charts from
//! successive polls. That is worse in three ways that all show up in use — a
//! reload throws the history away, a closed tab records nothing, and the widest
//! panels ("the last 60 minutes") cannot exist at all because the page has only
//! ever seen the last few minutes of its own uptime.
//!
//! zend keeps its history server-side for exactly these reasons
//! (`candle_conversation::scheduler::phase_ring`), and this is the same design:
//! trim by age with a length backstop, hand the whole window over on each poll.
//!
//! # Columns, not rows
//!
//! zend serialises an array of per-sample objects. This serialises one array
//! per field, which is smaller on the wire (no key repeated 1,800 times) and is
//! already the shape `makeChart` wants — `xs` plus a `data` array per series —
//! so the page does no transposing.
//!
//! It also makes absence cheap to state exactly. A field nothing has ever
//! measured serialises as `null`, one token, rather than 1,800 nulls; a field
//! measured only for part of the window is an array with `null` at the front.
//! The page can therefore tell "no engine" from "the engine started four
//! minutes ago" without being told separately.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Serialize;

use super::sample::Sample;

/// How much history to keep. The widest panel on the page is an hour, and an
/// hour of context is what turns "VRAM is at 80%" into "VRAM has been climbing
/// steadily for forty minutes".
pub const MAX_AGE: Duration = Duration::from_secs(60 * 60);

/// Length backstop, independent of the age rule. At the sampler's cadence an
/// hour is ~1,800 samples; this leaves room for a faster cadence without ever
/// letting the ring grow without bound if one is chosen.
pub const MAX_LEN: usize = 4096;

/// One field of the window, or `null` if nothing ever measured it.
///
/// Inner `None` is a sample where this particular field was unavailable —
/// a gap, which the page skips rather than plotting as zero.
pub type Column = Option<Vec<Option<f64>>>;

/// The window, as it goes onto the wire.
///
/// `t` is seconds since the oldest retained sample, so it ascends and the
/// newest lands at the right edge of the page's axis. Relative, not absolute:
/// the page never has to agree with the daemon about what time it is.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Series {
    pub t: Vec<f64>,

    // Measured by the driver and the OS — present whenever there is a card and
    // a machine, which is to say always, on the machines that run this.
    pub vram_total_mib: Column,
    pub vram_used_mib: Column,
    pub vram_free_mib: Column,
    pub host_total_mib: Column,
    pub host_used_mib: Column,
    pub rss_mib: Column,

    // Reported by the inference engine. `null` until there is one.
    pub weights_mib: Column,
    pub kv_mib: Column,
    pub image_mib: Column,
    pub decode_tps: Column,
    pub prefill_tps: Column,
    pub mean_npcs_per_decode: Column,
    pub max_batch: Column,
    pub npcs_active: Column,
    pub ticks_per_sec: Column,
    pub inbox_depth_p50: Column,
    pub inbox_depth_p99: Column,
    pub image_queue_depth: Column,
}

/// Bounded history of [`Sample`]s.
#[derive(Debug, Default)]
pub struct Ring {
    samples: VecDeque<Sample>,
}

impl Ring {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::new(),
        }
    }

    /// How many samples are retained. The trim rules are the interesting part
    /// of this type, and this is how they are asserted.
    ///
    /// No production caller: the serialiser walks `samples` directly. It is
    /// kept because the age and length bounds are this type's whole contract
    /// and there is no other way to state them in a test — deleting it would
    /// delete the coverage, not the code.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Add a sample and drop whatever has aged out.
    pub fn push(&mut self, s: Sample) {
        let now = s.at;
        self.samples.push_back(s);
        self.trim(now);
    }

    fn trim(&mut self, now: Instant) {
        let cutoff = now.checked_sub(MAX_AGE);
        while let Some(front) = self.samples.front() {
            let too_old = cutoff.is_some_and(|c| front.at < c);
            if too_old || self.samples.len() > MAX_LEN {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// The window as columns, timed relative to its own oldest sample.
    pub fn series(&self) -> Series {
        let s: Vec<&Sample> = self.samples.iter().collect();
        if s.is_empty() {
            return Series::default();
        }
        let origin = s[0].at;
        let t = s
            .iter()
            .map(|x| round1(x.at.saturating_duration_since(origin).as_secs_f64()))
            .collect();

        // MiB are whole numbers from the driver and the OS. Rounding them here
        // rather than shipping `1045.0000000000001` cuts the payload materially
        // at no cost to a chart whose pixels are coarser than a mebibyte.
        let mib = |f: &dyn Fn(&Sample) -> Option<f64>| column(&s, |x| f(x).map(f64::round));
        let rate = |f: &dyn Fn(&Sample) -> Option<f64>| column(&s, |x| f(x).map(round1));

        Series {
            t,
            vram_total_mib: mib(&|x| x.vram.total_mib.map(|v| v as f64)),
            vram_used_mib: mib(&|x| x.vram.used_mib.map(|v| v as f64)),
            vram_free_mib: mib(&|x| x.vram.free_mib.map(|v| v as f64)),
            host_total_mib: mib(&|x| x.host.total_mib.map(|v| v as f64)),
            host_used_mib: mib(&|x| x.host_used_mib()),
            rss_mib: mib(&|x| x.host.rss_mib.map(|v| v as f64)),

            weights_mib: mib(&|x| x.eng(|e| e.weights_mib)),
            kv_mib: mib(&|x| x.eng(|e| e.kv_mib)),
            image_mib: mib(&|x| x.eng(|e| e.image_mib)),
            decode_tps: rate(&|x| x.eng(|e| e.decode_tps)),
            prefill_tps: rate(&|x| x.eng(|e| e.prefill_tps)),
            mean_npcs_per_decode: rate(&|x| x.eng(|e| e.mean_npcs_per_decode)),
            max_batch: rate(&|x| x.eng(|e| e.max_batch)),
            npcs_active: rate(&|x| x.eng(|e| e.npcs_active)),
            ticks_per_sec: rate(&|x| x.eng(|e| e.ticks_per_sec)),
            inbox_depth_p50: rate(&|x| x.eng(|e| e.inbox_depth_p50)),
            inbox_depth_p99: rate(&|x| x.eng(|e| e.inbox_depth_p99)),
            image_queue_depth: rate(&|x| x.eng(|e| e.image_queue_depth)),
        }
    }

    /// The newest sample, for the cards that show a current value.
    ///
    /// Unused in production for the same reason as [`Ring::len`] — the reading
    /// the console shows comes from the serialised window — and kept for the
    /// same one: it is how a test says which sample won.
    #[allow(dead_code)]
    pub fn latest(&self) -> Option<&Sample> {
        self.samples.back()
    }
}

/// Collect one field across the window, collapsing "never measured" to `null`.
///
/// The collapse is what keeps an engine-free daemon's payload small, and it is
/// also a stronger statement than an array of nulls: the field was not measured
/// at any point in this window, full stop.
fn column(samples: &[&Sample], f: impl Fn(&Sample) -> Option<f64>) -> Column {
    let col: Vec<Option<f64>> = samples.iter().map(|s| f(s)).collect();
    if col.iter().all(Option::is_none) {
        None
    } else {
        Some(col)
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::device::{Host, Vram};
    use crate::telemetry::sample::Engine;

    fn at(base: Instant, secs: u64) -> Instant {
        base + Duration::from_secs(secs)
    }

    fn s(at: Instant, used: u64, engine: Option<Engine>) -> Sample {
        Sample {
            at,
            vram: Vram {
                total_mib: Some(24_576),
                used_mib: Some(used),
                free_mib: Some(24_576 - used),
                ..Default::default()
            },
            host: Host {
                total_mib: Some(65_457),
                free_mib: Some(40_000),
                rss_mib: Some(28),
            },
            engine,
        }
    }

    #[test]
    fn the_window_is_timed_from_its_own_oldest_sample() {
        let base = Instant::now();
        let mut r = Ring::new();
        for i in 0..4 {
            r.push(s(at(base, i * 2), 1_000 + i, None));
        }
        let ser = r.series();
        assert_eq!(ser.t, vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(
            ser.vram_used_mib.unwrap(),
            vec![Some(1_000.0), Some(1_001.0), Some(1_002.0), Some(1_003.0)]
        );
    }

    /// The reason the ring exists. A page reloaded at minute fifty still sees
    /// the fifty minutes before it, which no amount of client-side accumulation
    /// can provide.
    #[test]
    fn samples_older_than_the_window_are_dropped_and_the_rest_kept() {
        let base = Instant::now();
        let mut r = Ring::new();
        r.push(s(base, 100, None));
        r.push(s(at(base, 60), 200, None));
        assert_eq!(r.len(), 2);

        // A sample one second past the hour. The cutoff is an hour behind *it*,
        // so only the sample at `base` falls outside — the one at base+60 is
        // 3,541s old and belongs in the window.
        r.push(s(at(base, 3_601), 300, None));
        assert_eq!(
            r.len(),
            2,
            "trim is relative to the newest sample, not to `base`"
        );
        assert_eq!(r.latest().unwrap().vram.used_mib, Some(300));

        // Far enough ahead that both earlier samples age out.
        r.push(s(at(base, 3_700), 400, None));
        assert_eq!(r.len(), 2);
        let used = r.series().vram_used_mib.unwrap();
        assert_eq!(used, vec![Some(300.0), Some(400.0)]);
    }

    #[test]
    fn the_length_backstop_holds_even_inside_the_age_window() {
        let base = Instant::now();
        let mut r = Ring::new();
        // All within the hour, so only the length rule can bound this.
        for i in 0..(MAX_LEN + 50) {
            r.push(s(at(base, (i % 3_000) as u64), i as u64, None));
        }
        assert!(r.len() <= MAX_LEN + 1, "ring grew to {}", r.len());
    }

    /// A field nothing measured is one `null`, not a thousand of them — and
    /// that is a statement the page acts on, not just a size optimisation.
    #[test]
    fn a_never_measured_field_collapses_to_null() {
        let base = Instant::now();
        let mut r = Ring::new();
        for i in 0..3 {
            r.push(s(at(base, i), 1_000, None));
        }
        let ser = r.series();
        assert!(ser.decode_tps.is_none());
        assert!(ser.kv_mib.is_none());
        // Measured ones are still full arrays.
        assert_eq!(ser.vram_used_mib.unwrap().len(), 3);

        let json = serde_json::to_value(r.series()).unwrap();
        assert_eq!(json["decode_tps"], serde_json::Value::Null);
    }

    /// An engine that starts mid-window leaves a gap at the front. That has to
    /// survive as a gap: plotting it as zero would draw a throughput collapse
    /// that never happened.
    #[test]
    fn an_engine_starting_mid_window_leaves_a_leading_gap() {
        let base = Instant::now();
        let mut r = Ring::new();
        r.push(s(at(base, 0), 1_000, None));
        r.push(s(at(base, 2), 1_000, None));
        r.push(s(
            at(base, 4),
            1_000,
            Some(Engine {
                decode_tps: Some(41.5),
                ..Default::default()
            }),
        ));

        let col = r.series().decode_tps.expect("measured at least once");
        assert_eq!(col, vec![None, None, Some(41.5)]);
    }

    /// An engine reporting zero must not look like an engine that is not there.
    #[test]
    fn a_measured_zero_is_not_a_gap() {
        let base = Instant::now();
        let mut r = Ring::new();
        r.push(s(
            base,
            1_000,
            Some(Engine {
                decode_tps: Some(0.0),
                ..Default::default()
            }),
        ));
        assert_eq!(r.series().decode_tps.unwrap(), vec![Some(0.0)]);
    }

    #[test]
    fn an_empty_ring_serialises_without_panicking() {
        let ser = Ring::new().series();
        assert!(ser.t.is_empty());
        assert!(ser.vram_used_mib.is_none());
        serde_json::to_value(ser).unwrap();
    }
}
