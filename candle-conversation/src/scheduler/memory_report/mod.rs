//! The full-system memory snapshot: VRAM, KV arenas, the warm tier, host RAM
//! and the admission throttle, taken together.
//!
//! The scheduler publishes one report on its telemetry cadence. It is stored in
//! a process-wide slot, so the HTTP layer reads the latest without taking the
//! engine lock. Each publish is also written as one JSON debug line, so a run's
//! memory history can be rebuilt from the log alone. Accounting failures came
//! first in practice, so every quantity the throttles use appears side by side
//! in one place. `None` marks a source this build or run does not have, which
//! is different from a value of zero.
//!
//! - `build_report`: assembles the sections and the whole-card reconciliation.
//! - `sections_kv_host`: KV arenas, the warm tier, host RAM and admission.
//! - `sections_vram`: VRAM, held ground, weights, the gallery and the
//!   reconciliation.
//! - Tests: `tests_report`.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use candle::vram::{host_pinned_bytes, AllocClass, DriverModel};
use candle::wave_provenance::last_wave_declines;
use candle::Device;
use candle_nn::kv_cache::{global_arena_memory_report, SlotTenant};
use serde::Serialize;

use super::admission::admit_quantum;
use super::state_census::StateSection;
use super::Scheduler;
use crate::index_page_cache;

/// One full memory snapshot. All byte quantities are raw bytes; consumers
/// derive MiB. `None` sections mean the underlying source is unavailable in
/// this build/run (e.g. no CUDA governor), never that the value is zero.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryReport {
    /// Milliseconds since the Unix epoch at capture.
    pub captured_unix_ms: u64,
    pub vram: Option<VramSection>,
    pub kv: KvSection,
    pub warm: WarmSection,
    pub host: HostSection,
    pub admission: AdmissionSection,
    pub experts: ExpertSection,
    pub weights: WeightSection,
    pub gallery: GallerySection,
    /// The span's "state" band — its tenants — broken down by whose ground it
    /// is ([`super::state_census`]).
    pub state: StateSection,
    pub accounting: AccountingSection,
    /// The held ground, holder by holder. `None` before the reservation exists.
    pub held: Option<HeldSection>,
}

/// Latest published report plus its capture `Instant` (for age).
static LATEST: OnceLock<Mutex<Option<(MemoryReport, Instant)>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<(MemoryReport, Instant)>> {
    LATEST.get_or_init(|| Mutex::new(None))
}

/// Store `report` as the latest snapshot.
pub fn publish(report: MemoryReport) {
    *slot().lock().unwrap() = Some((report, Instant::now()));
}

/// The latest snapshot and its age in milliseconds, if one has been published.
pub fn latest() -> Option<(MemoryReport, u64)> {
    slot()
        .lock()
        .unwrap()
        .as_ref()
        .map(|(r, at)| (r.clone(), at.elapsed().as_millis() as u64))
}

mod build_report;
mod sections_kv_host;
mod sections_vram;
#[cfg(test)]
mod tests_report;

pub use sections_kv_host::*;
pub use sections_vram::*;
