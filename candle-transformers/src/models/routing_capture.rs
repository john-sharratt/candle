//! Routing-trace capture for offline MoE expert-predictor evaluation.
//!
//! The expert pipeline's speculative prefetch is driven by a transition-matrix
//! predictor ([`super::expert_lre`]).  Tuning that predictor against a live
//! 30B model run is slow — each iteration reloads the model and decodes on the
//! GPU.  This module decouples the two: a single instrumented run captures the
//! real per-layer routing decisions into a trace, which is then replayed
//! through the predictor offline (CPU-only, milliseconds) as many times as
//! needed.
//!
//! ## What is captured
//!
//! One [`RoutingRecord`] per MoE dispatch, in forward-pass order:
//!
//! - `config` — index of the test config that produced it (records from
//!   different configs are tagged so a trace can hold more than one).
//! - `pass`   — monotonic forward-pass counter within a config.  A new pass
//!   begins whenever the MoE layer index stops increasing (the same wrap test
//!   the predictor uses to reset its per-pass state).
//! - `layer`  — the MoE layer index (0-based among MoE layers).
//! - `experts` — the deduplicated set of expert IDs the router selected for
//!   this layer across all tokens in the dispatch (exactly what the predictor
//!   observes).
//! - `mass` — per-expert routing mass aligned to `experts`: the summed router
//!   weight of every token assigned to that expert this dispatch.  Lets the
//!   offline eval experiment with weight-aware observation.
//!
//! ## Gating
//!
//! Capture is inert unless explicitly enabled.  [`enable`] turns it on for a
//! target config; [`init_from_env`] enables it when `CANDLE_DUMP_ROUTING` is
//! set (the path is informational — the trace is drained via [`take`] by the
//! capturing test, which owns serialization).  When disabled, [`record`] is a
//! single relaxed atomic load.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

/// On-disk location of the checked-in routing-trace fixture (bincode + gzip).
///
/// Written by the focused capture test, read by the offline predictor eval.
pub const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/models/batch_test/fixtures/routing_trace_qwen3_30b.bin.gz"
);

/// One captured MoE dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRecord {
    /// Test-config index that produced this record.
    pub config: u16,
    /// Forward-pass counter within the config (one decode step = one pass).
    pub pass: u32,
    /// MoE layer index (0-based).
    pub layer: u16,
    /// Deduplicated expert IDs selected for this layer.
    pub experts: Vec<u32>,
    /// Per-expert routing mass, aligned to `experts`.
    pub mass: Vec<f32>,
}

/// Mutable capture state behind the global lock.
struct CaptureState {
    /// Whether records are being collected.
    enabled: bool,
    /// Config index whose records are retained (`None` = all configs).
    target_config: Option<u16>,
    /// Current config index (set by [`begin_config`]).
    config: u16,
    /// Current pass counter within the config.
    pass: u32,
    /// Last MoE layer index seen, to detect pass wraps.
    last_layer: Option<u16>,
    /// Collected records.
    records: Vec<RoutingRecord>,
}

impl CaptureState {
    const fn new() -> Self {
        Self {
            enabled: false,
            target_config: None,
            config: 0,
            pass: 0,
            last_layer: None,
            records: Vec::new(),
        }
    }
}

/// Fast-path gate read on the hot capture site without taking the lock.
static ENABLED: AtomicBool = AtomicBool::new(false);

fn state() -> &'static Mutex<CaptureState> {
    static STATE: OnceLock<Mutex<CaptureState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(CaptureState::new()))
}

/// Returns true if capture is currently active (single relaxed atomic load).
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Enable capture, retaining only records tagged with `target_config`.
///
/// Clears any previously collected records.
pub fn enable(target_config: u16) {
    if let Ok(mut s) = state().lock() {
        s.enabled = true;
        s.target_config = Some(target_config);
        s.config = 0;
        s.pass = 0;
        s.last_layer = None;
        s.records.clear();
    }
    ENABLED.store(true, Ordering::Relaxed);
}

/// Enable capture for *all* configs (records tagged by config index).
///
/// Clears any previously collected records.
pub fn enable_all() {
    if let Ok(mut s) = state().lock() {
        s.enabled = true;
        s.target_config = None;
        s.config = 0;
        s.pass = 0;
        s.last_layer = None;
        s.records.clear();
    }
    ENABLED.store(true, Ordering::Relaxed);
}

/// Enable capture from the environment.
///
/// When `CANDLE_DUMP_ROUTING` is set, capture turns on for the config named by
/// `CANDLE_DUMP_ROUTING_CONFIG` (default 1, the BF16×1 config in the standard
/// sweep).  Returns the output path if capture was enabled.
pub fn init_from_env() -> Option<String> {
    let path = std::env::var("CANDLE_DUMP_ROUTING").ok()?;
    let target = std::env::var("CANDLE_DUMP_ROUTING_CONFIG")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(1);
    enable(target);
    Some(path)
}

/// Disable capture (records are retained until [`take`]).
pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
    if let Ok(mut s) = state().lock() {
        s.enabled = false;
    }
}

/// Mark the start of a new test config.  Resets the pass counter.
pub fn begin_config(config: usize) {
    if let Ok(mut s) = state().lock() {
        s.config = config as u16;
        s.pass = 0;
        s.last_layer = None;
    }
}

/// Record one MoE dispatch.  No-op when capture is disabled or the current
/// config is not the retained target.
///
/// `layer` is the MoE layer index; `experts` the deduplicated active set;
/// `mass` the per-expert routing mass aligned to `experts`.
pub fn record(layer: usize, experts: &[usize], mass: &[f32]) {
    if !is_enabled() {
        return;
    }
    let Ok(mut s) = state().lock() else {
        return;
    };
    if !s.enabled {
        return;
    }
    if let Some(target) = s.target_config {
        if s.config != target {
            return;
        }
    }

    // A pass wraps when the layer index stops strictly increasing.
    let layer = layer as u16;
    match s.last_layer {
        Some(prev) if layer <= prev => s.pass += 1,
        _ => {}
    }
    s.last_layer = Some(layer);

    let config = s.config;
    let pass = s.pass;
    s.records.push(RoutingRecord {
        config,
        pass,
        layer,
        experts: experts.iter().map(|&e| e as u32).collect(),
        mass: mass.to_vec(),
    });
}

/// Drain and return all collected records, leaving the buffer empty.
pub fn take() -> Vec<RoutingRecord> {
    state()
        .lock()
        .map(|mut s| std::mem::take(&mut s.records))
        .unwrap_or_default()
}

/// Number of records currently buffered.
pub fn len() -> usize {
    state().lock().map(|s| s.records.len()).unwrap_or(0)
}
