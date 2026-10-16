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

pub mod bandwidth;
pub mod handoff;
pub mod pin;
pub mod shared_dram;
pub mod topology;
