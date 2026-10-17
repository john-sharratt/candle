//! The host CPU as a fourth place a routed expert is computed
//! (`docs/cpu_expert_compute.md`).
//!
//! | Module | Concern |
//! |---|---|
//! | [`topology`] | which logical processors are distinct physical cores, and which are performance cores |
//! | [`pin`] | keeping a worker thread on one of them |
//! | [`bandwidth`] | the DRAM read rate a set of pinned workers sustains — `B_d` in the split |
//! | [`shared_dram`] | that rate while the copy engine streams from the same DRAM — `B_c` |
//! | [`handoff`] | the device↔pool round trip through mapped words — `t_h` |
//! | [`kernel`] | the expert itself: q8 activations × KO weights, by ISA tier |
//! | [`pool`] | the pinned worker threads and their wake |
//! | `backoff` | how the cache's host threads wait: hot spins, low power, then sleep |
//! | [`layer`] | one layer's CPU experts on the pool: the dataflow schedule |
//! | [`spin_once`] | the once-cell a layer's tasks share an expert's input through, its waiters spinning |
//! | [`rings`] | the mapped work slot, row and result rings, doorbell, done and liveness words |
//! | [`service`] | the thread that serves the doorbell: rings in, layer on the pool, results out |
//! | `cold` | where the pool finds a cold expert: its pageable copy, or its publish into the pad |
//! | [`stats`] | where a served layer's time goes, by phase |
//! | [`params`] | `SplitParams`: the measured costs and the knobs, turned into the device's split |
//! | [`calibrate`] | the measured costs on this machine, at load |
//! | [`online`] | the measured costs following what the pool takes; late streaks reported |
//! | `attach` | starting the class for one expert cache: shapes, cores, rings, service |
//!
//! Measurements, `#[ignore]`d — run per machine:
//!
//! | Module | Measures |
//! |---|---|
//! | `kernel_profile` | the kernel's tuning harness: roof, hot and cold matvec, serial steps, pooled layer |
//! | `kernel_bench` | the layer at the split's count buckets — the `cpu_n_ns` table |
//! | `wait_bench` | the pool's waits on the decode pattern: start, join and done latency |
//!
//! Tests beside the code: `service_tests` (the service end to end, on a CUDA
//! context), `gpu_parity` (the pool's experts against the device's).

pub(crate) mod attach;
pub(crate) mod backoff;
pub mod bandwidth;
pub mod calibrate;
pub(crate) mod cold;
#[cfg(test)]
mod gpu_parity;
pub mod handoff;
pub mod kernel;
mod kernel_bench;
mod kernel_profile;
pub mod layer;
pub mod online;
pub mod params;
pub mod pin;
pub mod pool;
pub mod rings;
pub mod service;
#[cfg(test)]
mod service_tests;
pub mod shared_dram;
pub mod spin_once;
pub mod stats;
pub mod topology;
mod wait_bench;
