//! Where this process's host RAM is, by allocation.
//!
//! The machine-level counters say how much RAM is free; the pinned gauges say
//! how much this process page-locked. Neither says what the rest of the
//! process's resident memory *is* — and on a 31.5 GiB box that remainder
//! decides how large the warm expert tier may grow before the machine pages.
//! This walks the address space and reports every committed allocation with
//! its resident bytes, so the remainder can be named rather than inferred.

use std::collections::HashMap;

#[cfg(windows)]
mod windows;

/// What backs an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegionKind {
    /// A loaded executable image — the binary and its DLLs.
    Image,
    /// A file or section view (a memory-mapped file, a shared section).
    Mapped,
    /// Private memory: heaps, `VirtualAlloc`, driver host allocations.
    Private,
}

/// One committed region as the walk reports it — a run of pages sharing a
/// state inside one allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionRecord {
    /// The base of the allocation the region belongs to.
    pub alloc_base: u64,
    pub kind: RegionKind,
    pub committed: u64,
    /// Committed bytes currently in the process's working set.
    pub resident: u64,
}

/// One allocation: every committed region sharing an allocation base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    pub alloc_base: u64,
    pub kind: RegionKind,
    pub committed: u64,
    pub resident: u64,
    /// The backing file, for an image or a mapped view; `None` for private
    /// memory or when the name could not be read.
    pub name: Option<String>,
}

/// Committed and resident bytes of one [`RegionKind`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KindTotal {
    pub committed: u64,
    pub resident: u64,
}

/// Private allocations of committed size in `(upper / 2, upper]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeBucket {
    pub upper: u64,
    pub count: usize,
    pub committed: u64,
    pub resident: u64,
}

/// This process's committed host memory, allocation by allocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessRam {
    pub allocations: Vec<Allocation>,
}

impl ProcessRam {
    /// Walk the address space now. `None` where the platform has no walk.
    #[cfg(windows)]
    pub fn capture() -> Option<Self> {
        windows::capture()
    }

    /// See the Windows form.
    #[cfg(not(windows))]
    pub fn capture() -> Option<Self> {
        None
    }

    /// Group region records into allocations. Records of one allocation need
    /// not be adjacent; its kind is the first record's (an allocation has one
    /// type). Names are left empty for the caller to fill.
    pub fn from_records(records: impl IntoIterator<Item = RegionRecord>) -> Self {
        let mut allocations: Vec<Allocation> = Vec::new();
        let mut index = HashMap::new();
        for r in records {
            let i = *index.entry(r.alloc_base).or_insert_with(|| {
                allocations.push(Allocation {
                    alloc_base: r.alloc_base,
                    kind: r.kind,
                    committed: 0,
                    resident: 0,
                    name: None,
                });
                allocations.len() - 1
            });
            allocations[i].committed += r.committed;
            allocations[i].resident += r.resident;
        }
        Self { allocations }
    }

    /// Totals for `kind`.
    pub fn total(&self, kind: RegionKind) -> KindTotal {
        self.allocations
            .iter()
            .filter(|a| a.kind == kind)
            .fold(KindTotal::default(), |t, a| KindTotal {
                committed: t.committed + a.committed,
                resident: t.resident + a.resident,
            })
    }

    /// Private allocations bucketed by committed size — each bucket's upper
    /// bound (a power of two), how many allocations it holds, and their
    /// committed and resident bytes — smallest bucket first. Many small
    /// allocations hold resident memory that no single one explains; the
    /// bucket that carries it names the allocator.
    pub fn private_by_size(&self) -> Vec<SizeBucket> {
        let mut buckets: Vec<SizeBucket> = Vec::new();
        for a in self
            .allocations
            .iter()
            .filter(|a| a.kind == RegionKind::Private)
        {
            let upper = a.committed.max(1).next_power_of_two();
            match buckets.iter_mut().find(|b| b.upper == upper) {
                Some(b) => {
                    b.count += 1;
                    b.committed += a.committed;
                    b.resident += a.resident;
                }
                None => buckets.push(SizeBucket {
                    upper,
                    count: 1,
                    committed: a.committed,
                    resident: a.resident,
                }),
            }
        }
        buckets.sort_by_key(|b| b.upper);
        buckets
    }

    /// The `n` allocations holding the most resident bytes, largest first
    /// (ties by base address, so the order is stable).
    pub fn largest_resident(&self, n: usize) -> Vec<&Allocation> {
        let mut v: Vec<&Allocation> = self.allocations.iter().collect();
        v.sort_by(|a, b| {
            b.resident
                .cmp(&a.resident)
                .then(a.alloc_base.cmp(&b.alloc_base))
        });
        v.truncate(n);
        v
    }
}

#[cfg(test)]
mod tests;
