//! One reading of this process's committed host memory, as `/v1/memory` serves
//! it and the `host::ram` line logs it: totals by kind, the largest allocations
//! by committed bytes, and private memory bucketed by allocation size.

use candle::vram::process_ram::{Allocation, ProcessRam, RegionKind, SizeBucket};
use serde::Serialize;

/// How many of the largest allocations a reading keeps.
pub const LARGEST: usize = 32;

/// Committed and resident bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Bytes {
    pub committed: u64,
    pub resident: u64,
}

/// One allocation, named where the OS names its backing file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Held {
    /// Its base address, in hex: the same allocation in two readings has the
    /// same base.
    pub base: String,
    pub kind: &'static str,
    pub committed: u64,
    pub resident: u64,
    pub name: Option<String>,
}

/// Private allocations of committed size in `(upper / 2, upper]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Bucket {
    pub upper: u64,
    pub count: usize,
    pub committed: u64,
    pub resident: u64,
}

/// A reading, taken at `taken_unix_ms`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RamReading {
    pub taken_unix_ms: u64,
    pub private: Bytes,
    pub mapped: Bytes,
    pub image: Bytes,
    /// The [`LARGEST`] allocations by committed bytes, largest first.
    pub largest: Vec<Held>,
    /// Every private size bucket, smallest first.
    pub private_buckets: Vec<Bucket>,
}

fn kind_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Private => "private",
        RegionKind::Mapped => "mapped",
        RegionKind::Image => "image",
    }
}

fn held(a: &Allocation) -> Held {
    Held {
        base: format!("{:#x}", a.alloc_base),
        kind: kind_name(a.kind),
        committed: a.committed,
        resident: a.resident,
        name: a.name.clone(),
    }
}

fn bucket(b: &SizeBucket) -> Bucket {
    Bucket {
        upper: b.upper,
        count: b.count,
        committed: b.committed,
        resident: b.resident,
    }
}

impl RamReading {
    /// The reading of walk `ram`, taken at `taken_unix_ms`.
    pub fn of(ram: &ProcessRam, taken_unix_ms: u64) -> Self {
        let total = |kind| {
            let t = ram.total(kind);
            Bytes {
                committed: t.committed,
                resident: t.resident,
            }
        };
        Self {
            taken_unix_ms,
            private: total(RegionKind::Private),
            mapped: total(RegionKind::Mapped),
            image: total(RegionKind::Image),
            largest: ram
                .largest_committed(LARGEST)
                .into_iter()
                .map(held)
                .collect(),
            private_buckets: ram.private_by_size().iter().map(bucket).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use candle::vram::process_ram::RegionRecord;

    use super::*;

    const GIB: u64 = 1 << 30;

    fn record(alloc_base: u64, kind: RegionKind, committed: u64, resident: u64) -> RegionRecord {
        RegionRecord {
            alloc_base,
            kind,
            committed,
            resident,
        }
    }

    /// **A reading carries every figure of the walk**: the totals by kind, the
    /// allocations largest first with their bases in hex, and the private
    /// buckets smallest first.
    #[test]
    fn a_reading_carries_the_walk() {
        let mut ram = ProcessRam::from_records([
            record(0x2b4600000, RegionKind::Private, 58 * GIB, 58 * GIB),
            record(0x1000, RegionKind::Private, 3 << 20, 1 << 20),
            record(0x1000, RegionKind::Private, 1 << 20, 0),
            record(0x7ff7_0000_0000, RegionKind::Image, 400 << 20, 30 << 20),
            record(0x2ae7cda0000, RegionKind::Mapped, 2 * GIB, 0),
        ]);
        ram.allocations[2].name = Some("motherd.exe".into());
        let r = RamReading::of(&ram, 1_794_202_665_774);
        assert_eq!(r.taken_unix_ms, 1_794_202_665_774);
        assert_eq!(
            r.private,
            Bytes {
                committed: 58 * GIB + (4 << 20),
                resident: 58 * GIB + (1 << 20)
            }
        );
        assert_eq!(
            r.mapped,
            Bytes {
                committed: 2 * GIB,
                resident: 0
            }
        );
        assert_eq!(
            r.image,
            Bytes {
                committed: 400 << 20,
                resident: 30 << 20
            }
        );
        assert_eq!(
            r.largest,
            [
                Held {
                    base: "0x2b4600000".into(),
                    kind: "private",
                    committed: 58 * GIB,
                    resident: 58 * GIB,
                    name: None
                },
                Held {
                    base: "0x2ae7cda0000".into(),
                    kind: "mapped",
                    committed: 2 * GIB,
                    resident: 0,
                    name: None
                },
                Held {
                    base: "0x7ff700000000".into(),
                    kind: "image",
                    committed: 400 << 20,
                    resident: 30 << 20,
                    name: Some("motherd.exe".into())
                },
                Held {
                    base: "0x1000".into(),
                    kind: "private",
                    committed: 4 << 20,
                    resident: 1 << 20,
                    name: None
                },
            ]
        );
        assert_eq!(
            r.private_buckets,
            [
                Bucket {
                    upper: 4 << 20,
                    count: 1,
                    committed: 4 << 20,
                    resident: 1 << 20
                },
                Bucket {
                    upper: 64 * GIB,
                    count: 1,
                    committed: 58 * GIB,
                    resident: 58 * GIB
                },
            ]
        );
    }

    /// A reading keeps at most [`LARGEST`] allocations.
    #[test]
    fn a_reading_keeps_the_largest_only() {
        let ram = ProcessRam::from_records(
            (0..LARGEST as u64 + 5).map(|i| record(i << 20, RegionKind::Private, (i + 1) << 12, 0)),
        );
        let r = RamReading::of(&ram, 0);
        assert_eq!(r.largest.len(), LARGEST);
        assert_eq!(r.largest[0].committed, (LARGEST as u64 + 5) << 12);
    }
}
