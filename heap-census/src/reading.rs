//! One census reading: the census, the OS's figures, and when.

use super::classes::CLASSES;
use super::counting::{installed, CENSUS};
use super::os::{process_memory, OsHeap};

/// What the process held at one moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reading {
    /// When, in milliseconds on a **monotonic** clock: a window is the
    /// difference of two of these, and a wall clock stepped back would turn it
    /// negative.
    pub at_ms: u64,
    /// The commit charge.
    pub private_bytes: u64,
    /// The process heap's own figures, where the platform gives them.
    pub heap: Option<OsHeap>,
    /// Live bytes per size class from the counting allocator, when it is
    /// installed.
    pub classes: Option<[u64; CLASSES]>,
    /// Bytes ever allocated through the counting allocator, when installed.
    pub allocated_total: Option<u64>,
}

impl Reading {
    /// Live bytes our code holds: the census's, or else the heap's own count.
    pub fn live_bytes(&self) -> Option<u64> {
        match &self.classes {
            Some(c) => Some(c.iter().sum()),
            None => self.heap.map(|h| h.allocated_bytes),
        }
    }

    /// Bytes the heap has committed.
    pub fn heap_committed(&self) -> Option<u64> {
        self.heap.map(|h| h.committed_bytes)
    }

    /// Read now, at `at_ms`. `None` where the platform reads no process.
    pub fn take(at_ms: u64) -> Option<Reading> {
        let p = process_memory()?;
        let (classes, allocated_total) = if installed() {
            let n = CENSUS.counts();
            let mut c = [0u64; CLASSES];
            for (k, class) in n.classes.iter().enumerate() {
                c[k] = class.bytes;
            }
            (Some(c), Some(n.allocated_total))
        } else {
            (None, None)
        };
        Some(Reading {
            at_ms,
            private_bytes: p.private_bytes,
            heap: p.heap,
            classes,
            allocated_total,
        })
    }
}
