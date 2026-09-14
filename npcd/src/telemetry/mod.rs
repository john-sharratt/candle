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
    /// The startup load, when there is one — see [`Self::watch_loading`].
    loading: Mutex<Option<Arc<crate::engine::loading::LoadProgress>>>,
    model: ModelSpec,
}

impl Telemetry {
    pub fn new() -> Arc<Self> {
        let devices = Devices::open();
        // One model, not a ladder. It used to be chosen from the card's memory,
        // back when nothing loaded it — but the C-ladder KV thresholds are
        // derived against exactly one checkpoint, so a selection that varied by
        // card would silently vary the calibration with it.
        let model = model::spec();
        // The card is still worth logging beside the model even though it no
        // longer decides it: "7.5 GB of weights, 24 GiB of card" is the line an
        // operator reads to know whether this will fit before it tries.
        let (_, vram) = devices.sample_gpu();
        tracing::info!(
            "model: {} {} ({}, {:.1} GB) — {}",
            model.name,
            model.quant,
            model.params_total,
            model.bytes as f64 / 1e9,
            match vram.total_mib {
                Some(m) => format!("{:.1} GiB of card memory", m as f64 / 1024.0),
                None => "no card detected".to_owned(),
            }
        );
        // **Say so when this is not the repository's checkpoint.** An operator
        // reading a console that names a model they did not expect should learn
        // why from the log rather than by going looking for an override file. The
        // line is absent on a machine with no `models.override.yaml`, which is
        // the ordinary case.
        let overridden = candle_conversation::models::overrides::active_keys();
        if !overridden.is_empty() {
            tracing::info!(
                "models.override.yaml is in effect for {:?} — repo {} ({} adapter(s))",
                overridden,
                model.repo,
                crate::model::model().spec().loras.len()
            );
        }
        Arc::new(Self {
            devices,
            started: Instant::now(),
            ring: Mutex::new(Ring::new()),
            engine: Mutex::new(None),
            loading: Mutex::new(None),
            model,
        })
    }

    /// The same store, told when it started.
    ///
    /// Only the uptime differs, and only the test for uptime uses it: asserting
    /// that `uptime_s` counts real elapsed time otherwise means sleeping past a
    /// whole-second boundary, which cost 1.1 s — more than half the crate's
    /// entire test run — to check one subtraction.
    #[cfg(test)]
    fn started_at(started: Instant) -> Arc<Self> {
        Arc::new(Self {
            devices: Devices::open(),
            started,
            ring: Mutex::new(Ring::new()),
            engine: Mutex::new(None),
            loading: Mutex::new(None),
            model: model::spec(),
        })
    }

    /// The engine's hook: set the value the *next* sample will carry.
    ///
    /// A set rather than a push of its own sample — one timer owns the cadence,
    /// so an engine reporting at its own rhythm cannot bend the time axis every
    /// other panel shares.
    pub fn record(&self, e: Engine) {
        *self.engine.lock().unwrap() = Some(e);
    }

    /// Read the load progress on every sample, so a long ingest is visible on
    /// the performance page as it happens.
    ///
    /// **Startup is the one phase with heavy engine work and no engine to ask.**
    /// The `Engine` fields here were a stub, so a half-hour world load reported
    /// `prefill_tps: null` from beginning to end and the only way to know
    /// whether it was fast or slow was to poll `/v1/status` twice and do the
    /// arithmetic. `LoadProgress` already counts the tokens where they are
    /// sealed and derives the rate; this points the sampler at it rather than
    /// computing the same number a second way.
    ///
    /// Pulled per sample rather than pushed per document: an ingest seals
    /// several documents a second and the ring wants one reading every two.
    pub fn watch_loading(&self, loading: Arc<crate::engine::loading::LoadProgress>) {
        *self.loading.lock().unwrap() = Some(loading);
    }

    /// The engine reading for this instant: whatever an engine last reported,
    /// or — during startup — what the load progress can say.
    fn engine_now(&self) -> Option<Engine> {
        if let Some(e) = self.engine.lock().unwrap().clone() {
            return Some(e);
        }
        let l = self.loading.lock().unwrap().clone()?.snapshot()?;
        // Only once tokens have been counted. A phase that prefills nothing —
        // the model load, the calibration — has no rate to report, and a zero
        // there reads as a stalled engine rather than as an absent one.
        l.prefill_tps.map(|tps| Engine {
            prefill_tps: Some(tps),
            ..Engine::default()
        })
    }

    /// Take one sample and file it. Called by the sampler task, and directly by
    /// tests, which is why it is not buried inside the loop.
    pub fn tick(&self) {
        let (_, vram) = self.devices.sample_gpu();
        let s = Sample {
            at: Instant::now(),
            vram,
            host: device::sample_host(),
            engine: self.engine_now(),
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

    /// **A long ingest is visible while it runs.**
    ///
    /// The engine fields were a stub, so a half-hour world load reported
    /// `prefill_tps: null` from beginning to end — and startup is precisely the
    /// phase with heavy engine work and no engine to ask. The only way to know
    /// whether an ingest was fast or slow was to poll `/v1/status` twice and do
    /// the arithmetic by hand.
    #[test]
    fn an_ingest_reports_its_rate_before_the_engine_exists() {
        use crate::engine::loading::{LoadProgress, LoadStep};

        let loading = Arc::new(LoadProgress::new());
        let t = Telemetry::new();
        t.watch_loading(Arc::clone(&loading));

        // Nothing prefilled yet: absent, not zero. A zero here reads as a
        // stalled engine rather than as one that has not started.
        t.tick();
        assert!(t.read().series.prefill_tps.is_none());

        loading.set_step(LoadStep::Layers);
        loading.add_prefill_tokens(4_000);
        t.tick();

        let r = t.read();
        let series = r
            .series
            .prefill_tps
            .expect("the ingest rate never appeared");
        let latest = series.last().copied().flatten().expect("no reading");
        assert!(
            latest > 0.0,
            "the ingest reported a rate of {latest} tokens/s"
        );
        // **And the flag stays false.** `engine_connected` means *an engine has
        // reported*, and during startup none has — the loading counter has.
        // Flipping it here would tell the console the engine is up while every
        // route it would then call still answers 503. The rate belongs in the
        // series (and in `/v1/status`'s loading block); the flag keeps meaning
        // what it says.
        assert!(
            !r.engine_connected,
            "a loading daemon claimed a live engine because its ingest was measurable"
        );
    }

    /// A real engine's own reading wins over the load progress — once it is
    /// answering, it knows more than the startup counter does.
    #[test]
    fn a_live_engine_outranks_the_loading_counter() {
        use crate::engine::loading::{LoadProgress, LoadStep};

        let loading = Arc::new(LoadProgress::new());
        loading.set_step(LoadStep::Layers);
        loading.add_prefill_tokens(4_000);
        let t = Telemetry::new();
        t.watch_loading(loading);

        t.record(Engine {
            prefill_tps: Some(1234.0),
            ..Default::default()
        });
        t.tick();
        assert_eq!(
            t.read()
                .series
                .prefill_tps
                .unwrap()
                .last()
                .copied()
                .flatten(),
            Some(1234.0)
        );
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

    /// The model the console shows must be the one the loader fetches — a page
    /// naming a different checkpoint sends somebody looking for a bug in the
    /// engine.
    #[test]
    fn the_reported_model_is_the_one_this_daemon_runs() {
        let t = Telemetry::new();
        t.tick();
        let r = t.read();
        assert_eq!(r.model.repo, crate::model::spec().repo);

        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["model"]["name"], crate::model::spec().name);
        assert!(json["model"]["quant"].as_str().unwrap().starts_with('Q'));
        assert!(json["model"]["bytes"].as_u64().unwrap() > 0);
    }

    /// Uptime is the one thing that is always true, and it has to move.
    #[test]
    fn uptime_is_real() {
        // Wound back rather than waited out. `uptime_s` is
        // `started.elapsed().as_secs()`, so a store told it began 90 seconds ago
        // proves the same arithmetic a 1.1 s sleep did — and proves it harder,
        // because the exact number is asserted rather than "at least one".
        let now = Telemetry::new();
        assert_eq!(now.read().uptime_s, 0);

        let began = Instant::now()
            .checked_sub(Duration::from_secs(90))
            .expect("this machine has been up longer than the offset");
        assert_eq!(Telemetry::started_at(began).read().uptime_s, 90);
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
