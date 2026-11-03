//! The census as `/v1/memory` carries it under `process`.

use std::cmp::Reverse;

use serde::Serialize;

use super::classes::{upto_bytes, CLASSES};
use super::counting::{installed, CENSUS, LABELLED_FROM};
use super::os::{private_bytes, OsHeap};
use super::scope::Scope;
use super::shards::ClassCount;

/// One non-empty size class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ClassLine {
    pub class: usize,
    pub upto_bytes: u64,
    pub live_bytes: u64,
    pub live_count: u64,
}

/// The counting allocator's figures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AllocatorCounts {
    pub live_bytes: u64,
    pub live_count: u64,
    pub allocated_total: u64,
    pub freed_total: u64,
    /// The non-empty classes, smallest first.
    pub classes: Vec<ClassLine>,
    /// The smallest block that carries its scope: every scope line counts
    /// blocks of this many bytes or more.
    pub scoped_from_bytes: usize,
    /// The scopes holding anything, most bytes first.
    pub scopes: Vec<ScopeLine>,
}

/// What one scope's blocks hold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScopeLine {
    pub scope: &'static str,
    pub live_bytes: u64,
    pub live_count: u64,
    /// Its non-empty classes, smallest first.
    pub classes: Vec<ClassLine>,
}

/// The non-empty classes of `counts`, smallest first.
fn class_lines(counts: &[ClassCount; CLASSES]) -> Vec<ClassLine> {
    counts
        .iter()
        .enumerate()
        .filter(|(_, c)| c.count > 0 || c.bytes > 0)
        .map(|(k, c)| ClassLine {
            class: k,
            upto_bytes: upto_bytes(k),
            live_bytes: c.bytes,
            live_count: c.count,
        })
        .collect()
}

/// The process heap's own figures, as reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OsHeapLine {
    pub allocated_bytes: u64,
    pub committed_bytes: u64,
    /// Committed and not allocated: what the heap holds for nothing.
    pub free_committed_bytes: u64,
    pub read_us: u64,
}

impl From<OsHeap> for OsHeapLine {
    fn from(h: OsHeap) -> Self {
        OsHeapLine {
            allocated_bytes: h.allocated_bytes,
            committed_bytes: h.committed_bytes,
            free_committed_bytes: h.committed_bytes.saturating_sub(h.allocated_bytes),
            read_us: h.read_us,
        }
    }
}

/// The census, now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HeapReport {
    /// The commit charge, absent where the platform reads no process.
    pub private_bytes: Option<u64>,
    /// The process heap's own summary, where the platform gives one.
    pub os_heap: Option<OsHeapLine>,
    /// The counting allocator's figures; absent when it is not this
    /// process's global allocator.
    pub allocator: Option<AllocatorCounts>,
}

/// The allocator's live bytes and bytes ever allocated, when it is installed —
/// the two figures a frequent sampler wants, read without allocating, so the
/// sampler adds nothing to the churn it reports.
pub fn allocator_totals() -> Option<(u64, u64)> {
    if !installed() {
        return None;
    }
    let n = CENSUS.counts();
    Some((n.live_bytes(), n.allocated_total))
}

/// The counting allocator's figures now, when it is installed.
pub fn allocator_counts() -> Option<AllocatorCounts> {
    if !installed() {
        return None;
    }
    let n = CENSUS.counts();
    let mut scopes: Vec<ScopeLine> = Scope::ALL
        .iter()
        .zip(&n.scopes)
        .map(|(s, counts)| ScopeLine {
            scope: s.name(),
            live_bytes: counts.iter().map(|c| c.bytes).sum(),
            live_count: counts.iter().map(|c| c.count).sum(),
            classes: class_lines(counts),
        })
        .filter(|l| l.live_bytes > 0 || l.live_count > 0)
        .collect();
    scopes.sort_by_key(|l| Reverse(l.live_bytes));
    Some(AllocatorCounts {
        live_bytes: n.live_bytes(),
        live_count: n.live_count(),
        allocated_total: n.allocated_total,
        freed_total: n.freed_total(),
        classes: class_lines(&n.classes),
        scoped_from_bytes: LABELLED_FROM,
        scopes,
    })
}

/// The census now, with `os_heap` as the heap's summary. The summary holds the
/// heap's lock while it is read, so a caller polled often passes the one its
/// census task last took ([`crate::os::heap_summary`]) rather than reading it
/// per request; the commit charge and the allocator's counts are read fresh,
/// since neither locks anything.
pub fn report(os_heap: Option<OsHeap>) -> HeapReport {
    HeapReport {
        private_bytes: private_bytes(),
        os_heap: os_heap.map(OsHeapLine::from),
        allocator: allocator_counts(),
    }
}
