//! `GET /v1/memory` — the full memory accounting panel's data.
//!
//! The counterpart of zend's `zend/src/api/memory.rs`, and deliberately the
//! same shape: an engine section that is absent until there is an engine, plus
//! the two things only the HTTP moment can know — a fresh OS reading and this
//! process's own footprint.
//!
//! Those last two are not engine-dependent, which is the point. They are what
//! distinguishes *structurally* tight from *actively* short: a process whose
//! working set is small while the machine's available memory is low is not the
//! thing consuming it. That question is worth answering on a daemon with no
//! engine at all, which is why this endpoint is useful today rather than being
//! one more panel waiting on something.

use serde::Serialize;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

/// Response body for `GET /v1/memory`.
#[derive(Debug, Serialize)]
pub struct MemoryDump {
    /// The engine's own memory report — VRAM pools, KV arenas, warm tier.
    /// `null` until an engine publishes one, never a fabricated zero.
    pub report: Option<serde_json::Value>,
    /// Age of `report` in milliseconds, absent for the same reason.
    pub report_age_ms: Option<u64>,
    pub host_now: HostNow,
    pub process: ProcessNow,
}

/// OS memory at request time, in bytes.
#[derive(Debug, Serialize)]
pub struct HostNow {
    pub total_bytes: u64,
    /// Available = free + reclaimable. Higher than `free_bytes`, and the more
    /// honest of the two for "could something else run here".
    pub available_bytes: u64,
    pub free_bytes: u64,
}

/// This process at request time, in bytes.
#[derive(Debug, Serialize)]
pub struct ProcessNow {
    /// Resident working set — pages actually in RAM.
    pub working_set_bytes: u64,
    /// Virtual address-space size, not commit charge. Reported for completeness
    /// and easy to misread: a process that has merely *reserved* a large range
    /// has not taken that memory from anything.
    pub virtual_bytes: u64,
}

/// Read the OS and this process. A fresh `System` refreshed only for what is
/// asked — the default constructor enumerates every process on the machine,
/// which is a great deal of work to answer two questions.
pub fn dump() -> MemoryDump {
    let mut sys = System::new();
    sys.refresh_memory();

    let (working_set_bytes, virtual_bytes) = match sysinfo::get_current_pid() {
        Ok(pid) => {
            sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                ProcessRefreshKind::new().with_memory(),
            );
            sys.process(pid)
                .map(|p| (p.memory(), p.virtual_memory()))
                .unwrap_or((0, 0))
        }
        Err(_) => (0, 0),
    };

    MemoryDump {
        report: None,
        report_age_ms: None,
        host_now: HostNow {
            total_bytes: sys.total_memory(),
            available_bytes: sys.available_memory(),
            free_bytes: sys.free_memory(),
        },
        process: ProcessNow {
            working_set_bytes,
            virtual_bytes,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_machine_and_this_process_are_both_measurable() {
        let d = dump();
        assert!(d.host_now.total_bytes > 0, "a machine knows its own memory");
        assert!(d.host_now.available_bytes <= d.host_now.total_bytes);
        assert!(d.host_now.free_bytes <= d.host_now.available_bytes);
        // The test binary is a process, so it has a working set.
        assert!(d.process.working_set_bytes > 0);
    }

    /// The engine section stays absent rather than becoming an empty object —
    /// the page prints "not reported" from `null`, and `{}` would read as a
    /// report that happened to contain nothing.
    #[test]
    fn the_engine_report_is_absent_not_empty() {
        let json = serde_json::to_value(dump()).unwrap();
        assert_eq!(json["report"], serde_json::Value::Null);
        assert_eq!(json["report_age_ms"], serde_json::Value::Null);
        assert!(json["host_now"]["total_bytes"].as_u64().unwrap() > 0);
    }
}
