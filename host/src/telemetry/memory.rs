//! `GET /v1/memory` — the full memory accounting panel's data, for every tenant.
//!
//! The engine section is the shared scheduler's published snapshot
//! ([`candle_conversation::memory_report`], refreshed at the ~2 s wave cadence):
//! VRAM, KV arenas, warm tier, host RAM, admission throttle, expert residency.
//! It is absent until the engine has published one. Beside it, the two things
//! only the HTTP moment can know — a fresh OS reading and this process's own
//! footprint — which distinguish *structurally* tight from *actively* short: a
//! process whose working set is small while the machine's available memory is
//! low is not the thing consuming it.

use candle_conversation::memory_report::{latest, MemoryReport};
use candle_conversation::store_gauges::{all as store_gauges, StoreGauge};
use heap_census::report::{report as heap_report, HeapReport};
use serde::Serialize;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

use crate::memory_census::{latest as census_latest, CensusNow};
use crate::ram_census::{latest as ram_latest, RamReading};

/// Response body for `GET /v1/memory`.
#[derive(Serialize)]
pub struct MemoryDump {
    /// The engine's own memory report. `null` until the engine publishes one,
    /// never a fabricated zero.
    pub report: Option<MemoryReport>,
    /// Age of `report` in milliseconds, absent for the same reason.
    pub report_age_ms: Option<u64>,
    pub host_now: HostNow,
    pub process: ProcessNow,
    /// Each tenant's store, structure by structure, as its persistence pass
    /// last gauged it (`docs/memory_census.md` §5). Empty until each has been
    /// gauged, a minute after it opens.
    pub substrates: Vec<StoreGauge>,
}

/// OS memory at request time, in bytes.
#[derive(Serialize)]
pub struct HostNow {
    pub total_bytes: u64,
    /// Available = free + reclaimable. Higher than `free_bytes`, and the more
    /// honest of the two for "could something else run here".
    pub available_bytes: u64,
    pub free_bytes: u64,
}

/// This process at request time, in bytes.
#[derive(Serialize)]
pub struct ProcessNow {
    /// Resident working set — pages actually in RAM.
    pub working_set_bytes: u64,
    /// Virtual address-space size, not commit charge. Reported for completeness
    /// and easy to misread: a process that has merely *reserved* a large range
    /// has not taken that memory from anything.
    pub virtual_bytes: u64,
    /// The memory census (`docs/memory_census.md`): the commit charge, the
    /// heap's own summary as the census last read it, and the counting
    /// allocator's live bytes by size class.
    pub heap: HeapReport,
    /// The census's last reading: untracked memory, the machine's commit, and
    /// the growth verdict over the last half hour. Absent before the first
    /// reading, a minute after start.
    pub census: Option<CensusNow>,
    /// The address space's last walk (`ram_census`): committed and resident
    /// bytes by kind, the largest allocations and the private size buckets —
    /// what the commit charge is made of. Absent before the first walk.
    pub ram: Option<RamReading>,
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

    let (report, report_age_ms) = match latest() {
        Some((r, age)) => (Some(r), Some(age)),
        None => (None, None),
    };
    // The heap's summary as the census last read it: reading it here would
    // hold the heap's lock for every poll of this page.
    let census = census_latest();
    let heap = heap_report(census.as_ref().and_then(|c| c.os_heap));
    MemoryDump {
        report,
        report_age_ms,
        host_now: HostNow {
            total_bytes: sys.total_memory(),
            available_bytes: sys.available_memory(),
            free_bytes: sys.free_memory(),
        },
        process: ProcessNow {
            working_set_bytes,
            virtual_bytes,
            heap,
            census,
            ram: ram_latest().map(|r| RamReading::clone(&r)),
        },
        substrates: store_gauges(),
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

    /// **The census rides in the process block**: the commit charge always,
    /// and no allocator section in a binary that did not install the counting
    /// allocator — absent, not zero.
    #[test]
    #[cfg(any(windows, target_os = "linux"))]
    fn the_census_rides_in_the_process_block() {
        let json = serde_json::to_value(dump()).unwrap();
        assert!(json["process"]["heap"]["private_bytes"].as_u64().unwrap() > 0);
        assert_eq!(
            json["process"]["heap"]["allocator"],
            serde_json::Value::Null
        );
        // No census thread runs in a test binary: no walk yet, so no reading.
        assert_eq!(json["process"]["ram"], serde_json::Value::Null);
    }

    /// With no engine in this process the engine section stays absent rather
    /// than becoming an empty object — the page prints "not reported" from
    /// `null`, and `{}` would read as a report that happened to contain nothing.
    #[test]
    fn the_engine_report_is_absent_not_empty() {
        let json = serde_json::to_value(dump()).unwrap();
        assert_eq!(json["report"], serde_json::Value::Null);
        assert_eq!(json["report_age_ms"], serde_json::Value::Null);
        assert!(json["host_now"]["total_bytes"].as_u64().unwrap() > 0);
    }
}
