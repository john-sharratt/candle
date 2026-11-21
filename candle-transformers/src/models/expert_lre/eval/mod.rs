//! Offline replay evaluation of the MoE expert predictor, on a captured routing
//! trace and with no model or GPU.
//!
//! The module loads a trace captured by `routing_capture` and replays it through
//! `TransitionMatrix` on the CPU, so tuning the predictor takes milliseconds
//! rather than a model run. `Trace` groups capture records into passes keyed by
//! config and pass, keeps layer order canonical with the routing masses, and
//! offers per-config and leave-one-out sub-traces. The report runs from the cargo
//! test named in its doc, and does nothing until the fixture has been captured.
//!
//! - `metrics_replay`: replay and the precision and coverage metrics.
//! - `predictor`: the predictor trait and the candidate scoring rules.
//! - `session`: the multi-source and session-adaptive predictors.
//! - `recurrence_cache`: recurrence statistics and the cache simulator.
//! - `cache_outcome`: the end-to-end cache outcome at a VRAM budget.
//! - `demote`: group-eviction policies against LRU.
//! - `champion`: the champion predictor and its leave-one-out tuning.
//! - `loocv_ablations`: ablations of the session-prior predictor.
//! - `wave`: the batched decode wave and its token-rate model.
//! - `report`: the coverage report over a captured fixture.
//! - `tests_logic`: tests for the replay and metric logic.

// Test-only module (`#[cfg(test)] mod eval;`). The replay walks layer/expert
// index ranges the way the trace records them, so index loops are the shape of
// the data rather than a style choice.
#![allow(clippy::needless_range_loop, clippy::unnecessary_get_then_check)]

use std::collections::{HashMap, HashSet};
use std::io::Read;

use crate::models::routing_capture::{RoutingRecord, FIXTURE_PATH};

use super::cache::PINNED_LAYERS as PINNED;
use super::transition::TransitionMatrix;

/// A single forward pass: the per-layer active expert sets, in layer order.
struct Pass {
    /// Config index this pass belongs to (recurrence must not cross configs).
    config: u16,
    /// `(layer_idx, expert_ids)` for each MoE layer in this pass.
    layers: Vec<(usize, Vec<usize>)>,
    /// Per-layer routing mass aligned to `layers[i].1` (same order/length).
    masses: Vec<Vec<f32>>,
}

/// A replayable trace grouped into passes, plus inferred geometry.
struct Trace {
    passes: Vec<Pass>,
    num_layers: usize,
    num_experts: usize,
}

impl Trace {
    /// Group capture-ordered records into passes keyed by `(config, pass)`.
    fn from_records(records: &[RoutingRecord]) -> Self {
        let mut passes: Vec<Pass> = Vec::new();
        let mut cur_key: Option<(u16, u32)> = None;
        let mut num_layers = 0usize;
        let mut num_experts = 0usize;

        for r in records {
            let key = (r.config, r.pass);
            if cur_key != Some(key) {
                passes.push(Pass {
                    config: r.config,
                    layers: Vec::new(),
                    masses: Vec::new(),
                });
                cur_key = Some(key);
            }
            let experts: Vec<usize> = r.experts.iter().map(|&e| e as usize).collect();
            num_layers = num_layers.max(r.layer as usize + 1);
            if let Some(&m) = r.experts.iter().max() {
                num_experts = num_experts.max(m as usize + 1);
            }
            let p = passes.last_mut().unwrap();
            p.layers.push((r.layer as usize, experts));
            p.masses.push(r.mass.clone());
        }

        // Keep layer order canonical within each pass (layers + masses together).
        for p in &mut passes {
            let mut idx: Vec<usize> = (0..p.layers.len()).collect();
            idx.sort_by_key(|&i| p.layers[i].0);
            p.layers = idx.iter().map(|&i| p.layers[i].clone()).collect();
            p.masses = idx.iter().map(|&i| p.masses[i].clone()).collect();
        }

        Trace {
            passes,
            num_layers,
            num_experts,
        }
    }

    /// Average active-expert-set size across all (pass, layer) records.
    fn avg_set_size(&self) -> f64 {
        let (mut n, mut sum) = (0usize, 0usize);
        for p in &self.passes {
            for (_, e) in &p.layers {
                n += 1;
                sum += e.len();
            }
        }
        if n == 0 {
            0.0
        } else {
            sum as f64 / n as f64
        }
    }

    /// Sorted list of distinct config (prompt) indices present in the trace.
    fn configs(&self) -> Vec<u16> {
        let mut cs: Vec<u16> = self.passes.iter().map(|p| p.config).collect();
        cs.sort_unstable();
        cs.dedup();
        cs
    }

    /// A sub-trace containing only the passes of one config (prompt).
    fn for_config(&self, config: u16) -> Trace {
        self.filter(|p| p.config == config)
    }

    /// A sub-trace of every config *except* `config` (the LOOCV training set).
    fn for_configs_except(&self, config: u16) -> Trace {
        self.filter(|p| p.config != config)
    }

    fn filter(&self, keep: impl Fn(&Pass) -> bool) -> Trace {
        let passes = self
            .passes
            .iter()
            .filter(|p| keep(p))
            .map(|p| Pass {
                config: p.config,
                layers: p.layers.clone(),
                masses: p.masses.clone(),
            })
            .collect();
        Trace {
            passes,
            num_layers: self.num_layers,
            num_experts: self.num_experts,
        }
    }
}

// ── Deterministic tests of the replay/metric logic (no fixture needed) ──

mod cache_outcome;
mod champion;
mod demote;
mod loocv_ablations;
mod metrics_replay;
mod predictor;
mod recurrence_cache;
mod report;
mod session;
#[cfg(test)]
mod tests_logic;
mod wave;

use champion::*;
use loocv_ablations::*;
use metrics_replay::*;
use predictor::*;
use recurrence_cache::*;
use report::*;
use session::*;
