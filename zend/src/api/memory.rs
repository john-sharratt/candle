//! `GET /v1/memory` — full memory dump: VRAM, KV arenas, warm tier, host RAM,
//! admission throttle, expert residency.
//!
//! The heavy sections come from the scheduler's published snapshot
//! ([`candle_conversation::memory_report`], refreshed at the ~2 s wave
//! cadence and also logged as one JSON debug line per publish). This handler
//! adds what only the HTTP moment can know: a fresh OS memory reading and this
//! process's own working set / commit — the numbers that distinguish
//! "structurally tight" (pinned pools, WDDM commit backing) from "actively
//! short". Nothing here takes the engine lock.

use axum::Json;
use candle_conversation::memory_report::MemoryReport;
use serde::Serialize;
use sysinfo::{ProcessesToUpdate, System};

/// Response body for `GET /v1/memory`.
#[derive(Serialize)]
pub struct MemoryDump {
    /// The scheduler's latest snapshot. `null` until the first wave flush
    /// (~2 s after the engine starts).
    pub report: Option<MemoryReport>,
    /// Age of `report` in milliseconds.
    pub report_age_ms: Option<u64>,
    /// OS memory measured at request time (not the scheduler's cached probe).
    pub host_now: HostNow,
    /// This process, measured at request time.
    pub process: ProcessNow,
}

/// Request-time OS memory reading.
#[derive(Serialize)]
pub struct HostNow {
    pub total_bytes: u64,
    /// Available = free + standby + zeroed; the number the ingest floor gates on.
    pub available_bytes: u64,
    pub free_bytes: u64,
}

/// Request-time view of the zend process itself.
#[derive(Serialize)]
pub struct ProcessNow {
    /// Resident working set — the process pages actually in RAM. Far below the
    /// system commit numbers is the WDDM signature: committed GPU backing that
    /// is paged out, not live RAM.
    pub working_set_bytes: u64,
    /// Virtual address-space size (`VirtualSize`) — NOT commit charge. CUDA
    /// reserves enormous VA ranges, so this exceeds even the system-wide commit
    /// total; it is reported for completeness only. The real commit numbers are
    /// `report.host.commit_total_bytes` / `commit_limit_bytes` (syscall-derived,
    /// system-wide).
    pub virtual_bytes: u64,
}

pub async fn dump() -> Json<MemoryDump> {
    let latest = candle_conversation::memory_report::latest();
    let (report, report_age_ms) = match latest {
        Some((r, age)) => (Some(r), Some(age)),
        None => (None, None),
    };

    let mut sys = System::new();
    sys.refresh_memory();
    let host_now = HostNow {
        total_bytes: sys.total_memory(),
        available_bytes: sys.available_memory(),
        free_bytes: sys.free_memory(),
    };

    let process = match sysinfo::get_current_pid() {
        Ok(pid) => {
            sys.refresh_processes(ProcessesToUpdate::Some(&[pid]));
            sys.process(pid)
                .map(|p| ProcessNow {
                    working_set_bytes: p.memory(),
                    virtual_bytes: p.virtual_memory(),
                })
                .unwrap_or(ProcessNow {
                    working_set_bytes: 0,
                    virtual_bytes: 0,
                })
        }
        Err(_) => ProcessNow {
            working_set_bytes: 0,
            virtual_bytes: 0,
        },
    };

    Json(MemoryDump {
        report,
        report_age_ms,
        host_now,
        process,
    })
}
