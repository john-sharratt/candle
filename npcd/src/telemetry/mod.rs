//! `/v1/telemetry` and `/v1/memory` — what this daemon can say about itself.
//!
//! Both used to fall through to `web::mock::npcd`, which answered with
//! fixtures: a `mock device`, a mean batch of 2.4, a throughput that never
//! moved. Those read as measurements on a page whose entire purpose is
//! measurement.
//!
//! # Measured, or absent — never plausible
//!
//! Everything here is optional, and the distinction is load-bearing. The card,
//! its memory and the host are real; everything downstream of the inference
//! engine is absent, because there is no engine yet and a zero would be a
//! claim. "Nothing is running" and "there is nothing to ask" are different
//! facts, and the console draws them differently.
//!
//! # A history, kept here
//!
//! [`ring`] holds an hour of samples, taken on a timer rather than only when
//! somebody is looking. That is what lets the page open with an hour of context
//! already drawn, survive a reload, and record what happened while the tab was
//! shut — none of which client-side accumulation can do. See that module for
//! the argument in full; it is the same design zend uses, for the same reasons.
//!
//! When the engine arrives it calls [`Telemetry::record`] and every panel fills
//! in. Nothing here needs revisiting to make that happen.

pub mod device;
pub mod memory;
pub mod ring;
pub mod sample;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::model::{self, ModelSpec};
use device::{Devices, Gpu, Host};
use ring::{Ring, Series};
use sample::{Engine, Sample};

/// How often a sample is taken. Matches zend's wave cadence, which is what the
/// panel widths on the performance page were drawn against; an hour of it is
/// ~1,800 samples.
pub const SAMPLE_PERIOD: Duration = Duration::from_secs(2);

/// One poll's answer.
#[derive(Debug, Clone, Serialize)]
pub struct Reading {
    /// What the card is. Identity, not a measurement, so it sits outside the
    /// series.
    pub gpu: Gpu,
    /// Which model this card gets. Selected from its memory at startup — see
    /// [`crate::model`] — not loaded, which is why the console labels it a
    /// selection until `engine_connected` says otherwise.
    pub model: ModelSpec,
    /// The host right now, stamped at request time rather than read from the
    /// newest sample — up to a sampling period fresher, and free.
    pub host: Host,
    /// The window. Every chart on the page reads from here.
    pub series: Series,
    /// Seconds between samples, so the page can label its axis without
    /// inferring the cadence from timestamps.
    pub sample_period_s: f64,
    /// Whether an engine has ever reported. The console shows *not measured*
    /// rather than zeroes when this is false, so an idle engine and an absent
    /// one cannot be confused.
    pub engine_connected: bool,
    /// The image queue's current state — a label, not a series.
    pub image_queue_state: Option<String>,
    /// Seconds this daemon has been up: the one number that is always true, and
    /// the first thing worth knowing when a page looks wrong.
    pub uptime_s: u64,
}

/// The live store: a driver handle, an hour of history, and whatever the engine
/// last said.
pub struct Telemetry {
    devices: Devices,
    started: Instant,
    ring: Mutex<Ring>,
    engine: Mutex<Option<Engine>>,
    model: ModelSpec,
}

impl Telemetry {
    pub fn new() -> Arc<Self> {
        let devices = Devices::open();
        // Decided once, from the card that is actually here. A card does not
        // grow memory while the process runs, so re-deciding per request would
        // only be a chance for two requests to disagree.
        let (_, vram) = devices.sample_gpu();
        let total_bytes = vram.total_mib.map(|m| m * 1024 * 1024);
        let model = model::choose(total_bytes);
        tracing::info!(
            "model selected: {} {} ({} total / {} active, {:.1} GB) — {}",
            model.name,
            model.quant,
            model.params_total,
            model.params_active,
            model.bytes as f64 / 1e9,
            match vram.total_mib {
                Some(m) => format!("{:.1} GiB of card memory", m as f64 / 1024.0),
                None => "no card detected".to_owned(),
            }
        );
        Arc::new(Self {
            devices,
            started: Instant::now(),
            ring: Mutex::new(Ring::new()),
            engine: Mutex::new(None),
            model,
        })
    }

    /// The engine's hook. Nothing calls this yet; when something does, the page
    /// stops saying *not measured* on its own.
    ///
    /// This sets the value the *next* sample will carry rather than pushing a
    /// sample of its own. One timer owns the cadence, so an engine reporting at
    /// its own rhythm cannot bend the time axis every other panel shares.
    ///
    /// Allowed dead because it is the one entry point an engine calls and the
    /// tests below drive the whole "absent until reported, then present"
    /// behaviour through it. Deleting it to silence a warning would delete that
    /// behaviour's only description.
    #[allow(dead_code)]
    pub fn record(&self, e: Engine) {
        *self.engine.lock().unwrap() = Some(e);
    }

    /// Take one sample and file it. Called by the sampler task, and directly by
    /// tests, which is why it is not buried inside the loop.
    pub fn tick(&self) {
        let (_, vram) = self.devices.sample_gpu();
        let s = Sample {
            at: Instant::now(),
            vram,
            host: device::sample_host(),
            engine: self.engine.lock().unwrap().clone(),
        };
        self.ring.lock().unwrap().push(s);
    }

    /// Run the sampler for the life of the process.
    ///
    /// Sampling on a timer rather than on request is the whole point: a page
    /// nobody has open still accumulates the history that page will want, and
    /// two consoles polling at different rates see one consistent series
    /// instead of each perturbing it.
    pub fn spawn_sampler(self: &Arc<Self>) {
        let me = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(SAMPLE_PERIOD);
            // A sampler that fell behind must not then fire a burst of
            // back-to-back samples with near-identical timestamps; skipping the
            // missed beats keeps the series evenly spaced.
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                // Driver and OS reads are blocking, and short. Doing them on a
                // blocking thread keeps a slow NVML query off the async workers
                // that are serving the console.
                let t = Arc::clone(&me);
                if tokio::task::spawn_blocking(move || t.tick()).await.is_err() {
                    break;
                }
            }
        });
    }

    pub fn read(&self) -> Reading {
        let (gpu, _) = self.devices.sample_gpu();
        let ring = self.ring.lock().unwrap();
        let engine = self.engine.lock().unwrap().clone();
        Reading {
            gpu,
            model: self.model,
            host: device::sample_host(),
            series: ring.series(),
            sample_period_s: SAMPLE_PERIOD.as_secs_f64(),
            engine_connected: engine.is_some(),
            image_queue_state: engine.and_then(|e| e.image_queue_state),
            uptime_s: self.started.elapsed().as_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no engine, every engine column is absent and the flag says so —
    /// which is what stops the console drawing zeroes that look like readings.
    #[test]
    fn without_an_engine_nothing_is_claimed() {
        let t = Telemetry::new();
        t.tick();
        let r = t.read();

        assert!(!r.engine_connected);
        assert!(r.series.decode_tps.is_none());
        assert!(r.series.mean_npcs_per_decode.is_none());
        assert!(r.series.kv_mib.is_none());
        assert!(r.image_queue_state.is_none());

        // The host is real regardless, so the page is never entirely empty.
        assert!(r.host.total_mib.is_some_and(|m| m > 0));
        assert_eq!(r.series.t.len(), 1);
        assert!(r.series.host_used_mib.is_some());
    }

    /// The behaviour the ring exists for: history accrues across polls, so a
    /// page arriving late still has something to draw.
    #[test]
    fn history_accumulates_independently_of_who_is_looking() {
        let t = Telemetry::new();
        for _ in 0..5 {
            t.tick();
        }
        assert_eq!(t.read().series.t.len(), 5);
    }

    #[test]
    fn an_engine_reading_reaches_the_series_and_flips_the_flag() {
        let t = Telemetry::new();
        t.tick();
        assert!(!t.read().engine_connected);

        t.record(Engine {
            decode_tps: Some(41.5),
            npcs_active: Some(7.0),
            image_queue_state: Some("waiting_for_vram".to_owned()),
            ..Default::default()
        });
        t.tick();

        let r = t.read();
        assert!(r.engine_connected);
        assert_eq!(r.image_queue_state.as_deref(), Some("waiting_for_vram"));
        // The sample taken before the engine reported stays a gap, not a zero.
        assert_eq!(r.series.decode_tps.unwrap(), vec![None, Some(41.5)]);
        // A field the engine did not fill is still absent entirely.
        assert!(r.series.prefill_tps.is_none());
    }

    /// `record` must not itself add a sample — one timer owns the cadence, or
    /// the time axis bends whenever the engine reports at its own rhythm.
    #[test]
    fn recording_does_not_add_a_sample() {
        let t = Telemetry::new();
        t.tick();
        t.record(Engine::default());
        t.record(Engine::default());
        assert_eq!(t.read().series.t.len(), 1);
    }

    /// The selection has to follow the card that is actually present, and it
    /// has to reach the wire — a console showing the wrong quant would send
    /// somebody looking for a bug in the engine.
    #[test]
    fn the_model_matches_the_card_this_machine_has() {
        let t = Telemetry::new();
        t.tick();
        let r = t.read();

        let expected = crate::model::choose(
            r.series
                .vram_total_mib
                .as_ref()
                .and_then(|c| c.last().copied().flatten())
                .map(|m| m as u64 * 1024 * 1024),
        );
        assert_eq!(r.model, expected);

        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["model"]["name"], "Qwen3-30B-A3B");
        assert!(json["model"]["quant"].as_str().unwrap().starts_with('Q'));
        assert!(json["model"]["bytes"].as_u64().unwrap() > 0);
    }

    /// Uptime is the one thing that is always true, and it has to move.
    #[test]
    fn uptime_is_real() {
        let t = Telemetry::new();
        assert_eq!(t.read().uptime_s, 0);
        std::thread::sleep(Duration::from_millis(1_100));
        assert!(t.read().uptime_s >= 1);
    }

    /// The sampler must actually run without anybody asking it to.
    #[tokio::test(start_paused = true)]
    async fn the_sampler_fills_the_ring_on_its_own() {
        let t = Telemetry::new();
        t.spawn_sampler();
        assert_eq!(t.read().series.t.len(), 0);

        // Paused time: this advances the clock rather than waiting on it.
        tokio::time::sleep(SAMPLE_PERIOD * 4 + Duration::from_millis(100)).await;
        // Let the spawn_blocking hops land.
        tokio::task::yield_now().await;

        assert!(
            t.read().series.t.len() >= 3,
            "sampler produced {} samples",
            t.read().series.t.len()
        );
    }
}
