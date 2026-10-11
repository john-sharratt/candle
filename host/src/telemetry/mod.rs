//! `/v1/telemetry` and `/v1/memory` — what this process can say about itself.
//!
//! # Measured, or absent — never plausible
//!
//! Everything here is optional, and the distinction is load-bearing. The card,
//! its memory and the host are always real; the engine's columns are absent
//! until the scheduler has run a wave, because a zero would be a claim. "Nothing
//! is running" and "there is nothing to ask" are different facts, and the
//! consoles draw them differently.
//!
//! # A history, kept here
//!
//! [`ring`] holds an hour of samples, taken on a timer rather than only when
//! somebody is looking. That is what lets a page open with an hour of context
//! already drawn, survive a reload, and record what happened while the tab was
//! shut — none of which client-side accumulation can do.
//!
//! # Whose columns
//!
//! The engine's — decode and prefill rates, KV and weights — are read off the
//! shared scheduler ([`engine_reading`]), so they describe every tenant's work
//! together. A tenant's own columns (npcd's characters, inboxes and image
//! queue) are contributed by that tenant through [`Telemetry::add_source`] and
//! share the same time axis.

pub mod device;
pub mod engine_reading;
pub mod memory;
pub mod ring;
pub mod sample;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::time::{interval, MissedTickBehavior};

use crate::model::ModelView;
use device::{Devices, Gpu, Host};
use ring::{Ring, Series};
use sample::{Engine, Sample};

/// How often a sample is taken. Matches the scheduler's wave cadence, which is
/// what the panel widths on the performance pages were drawn against; an hour
/// of it is ~1,800 samples.
pub const SAMPLE_PERIOD: Duration = Duration::from_secs(2);

/// A tenant's contribution to a sample: it sets the columns it measures.
pub type Source = Arc<dyn Fn(&mut Engine) + Send + Sync>;

/// One poll's answer.
#[derive(Debug, Clone, Serialize)]
pub struct Reading {
    /// What the card is. Identity, not a measurement, so it sits outside the
    /// series.
    pub gpu: Gpu,
    /// The model every tenant runs.
    pub model: ModelView,
    /// The host right now, stamped at request time rather than read from the
    /// newest sample — up to a sampling period fresher, and free.
    pub host: Host,
    /// The window. Every chart on the page reads from here.
    pub series: Series,
    /// Seconds between samples, so the page can label its axis without
    /// inferring the cadence from timestamps.
    pub sample_period_s: f64,
    /// Whether the engine has run a wave yet — whether its columns hold a
    /// measurement. The consoles show *not measured* rather than zeroes when
    /// this is false, so an idle engine and an absent one cannot be confused.
    /// (`/v1/status`'s `engine_connected` is the other question: is the model
    /// loaded.)
    pub engine_measured: bool,
    /// The image queue's current state — a label, not a series.
    pub image_queue_state: Option<String>,
    /// Seconds this process has been up: the one number that is always true,
    /// and the first thing worth knowing when a page looks wrong.
    pub uptime_s: u64,
}

/// The live store: a driver handle, an hour of history, and the tenants'
/// contributions.
pub struct Telemetry {
    devices: Devices,
    started: Instant,
    ring: Mutex<Ring>,
    sources: Mutex<Vec<Source>>,
    /// The newest sample's engine reading, for the fields that are not series.
    latest: Mutex<Option<Engine>>,
    model: ModelView,
}

impl Telemetry {
    pub fn new(model: ModelView) -> Arc<Self> {
        Self::started_at(model, Instant::now())
    }

    /// The same store, told when it started — which lets the uptime test
    /// assert an exact number instead of sleeping past a second boundary.
    fn started_at(model: ModelView, started: Instant) -> Arc<Self> {
        let devices = Devices::open();
        let (_, vram) = devices.sample_gpu();
        tracing::info!(
            "model: {} ({}) — {}",
            model.name,
            match model.bytes {
                Some(b) => format!("{:.1} GB", b as f64 / 1e9),
                None => "prepared locally".to_owned(),
            },
            match vram.total_mib {
                Some(m) => format!("{:.1} GiB of card memory", m as f64 / 1024.0),
                None => "no card detected".to_owned(),
            }
        );
        Arc::new(Self {
            devices,
            started,
            ring: Mutex::new(Ring::new()),
            sources: Mutex::new(Vec::new()),
            latest: Mutex::new(None),
            model,
        })
    }

    /// Add a tenant's columns to every sample from now on.
    pub fn add_source(&self, source: Source) {
        self.sources.lock().unwrap().push(source);
    }

    /// The engine reading for this instant: the scheduler's, with each
    /// tenant's columns set on it. Absent with no engine.
    fn engine_now(&self) -> Option<Engine> {
        let mut e = engine_reading::engine_now(SAMPLE_PERIOD.as_secs_f64())?;
        for source in self.sources.lock().unwrap().iter() {
            source(&mut e);
        }
        Some(e)
    }

    /// Take one sample and file it. Called by the sampler task, and directly by
    /// tests, which is why it is not buried inside the loop.
    pub fn tick(&self) {
        let (_, vram) = self.devices.sample_gpu();
        let engine = self.engine_now();
        *self.latest.lock().unwrap() = engine.clone();
        let s = Sample {
            at: Instant::now(),
            vram,
            host: device::sample_host(),
            engine,
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
            let mut tick = interval(SAMPLE_PERIOD);
            // A sampler that fell behind must not then fire a burst of
            // back-to-back samples with near-identical timestamps; skipping the
            // missed beats keeps the series evenly spaced.
            tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                // Driver and OS reads are blocking, and short. Doing them on a
                // blocking thread keeps a slow NVML query off the async workers
                // that are serving the consoles.
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
        let latest = self.latest.lock().unwrap().clone();
        Reading {
            gpu,
            model: self.model.clone(),
            host: device::sample_host(),
            series: ring.series(),
            sample_period_s: SAMPLE_PERIOD.as_secs_f64(),
            engine_measured: latest.is_some(),
            image_queue_state: latest.and_then(|e| e.image_queue_state),
            uptime_s: self.started.elapsed().as_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> ModelView {
        ModelView::of_prescribed()
    }

    /// The host is real regardless, so a page is never entirely empty; the
    /// series grows one sample per tick.
    #[test]
    fn the_host_is_always_measured_and_history_accumulates() {
        let t = Telemetry::new(view());
        for _ in 0..5 {
            t.tick();
        }
        let r = t.read();
        assert!(r.host.total_mib.is_some_and(|m| m > 0));
        assert_eq!(r.series.t.len(), 5);
        assert!(r.series.host_used_mib.is_some());
    }

    /// The model a console shows is the one the process runs.
    #[test]
    fn the_reported_model_is_the_prescribed_one() {
        let t = Telemetry::new(view());
        let json = serde_json::to_value(t.read()).unwrap();
        assert_eq!(json["model"]["repo"], view().repo);
        assert_eq!(json["model"]["name"], view().name);
        assert_eq!(json["model"]["bytes"].as_u64(), view().bytes);
    }

    /// Uptime is the one thing that is always true, and it has to move.
    #[test]
    fn uptime_is_real() {
        assert_eq!(Telemetry::new(view()).read().uptime_s, 0);
        let began = Instant::now()
            .checked_sub(Duration::from_secs(90))
            .expect("this machine has been up longer than the offset");
        assert_eq!(Telemetry::started_at(view(), began).read().uptime_s, 90);
    }

    /// The sampler runs without anybody asking it to.
    #[tokio::test(start_paused = true)]
    async fn the_sampler_fills_the_ring_on_its_own() {
        let t = Telemetry::new(view());
        t.spawn_sampler();
        assert_eq!(t.read().series.t.len(), 0);
        tokio::time::sleep(SAMPLE_PERIOD * 4 + Duration::from_millis(100)).await;
        tokio::task::yield_now().await;
        assert!(
            t.read().series.t.len() >= 3,
            "sampler produced {} samples",
            t.read().series.t.len()
        );
    }
}
