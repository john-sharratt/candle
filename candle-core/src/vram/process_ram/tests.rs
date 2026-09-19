use super::*;

fn rec(alloc_base: u64, kind: RegionKind, committed: u64, resident: u64) -> RegionRecord {
    RegionRecord {
        alloc_base,
        kind,
        committed,
        resident,
    }
}

fn sample() -> ProcessRam {
    ProcessRam::from_records([
        rec(0x1000, RegionKind::Image, 4096, 4096),
        rec(0x9000, RegionKind::Private, 1 << 30, 3 << 20),
        rec(0x1000, RegionKind::Image, 8192, 0),
        rec(0x7000, RegionKind::Mapped, 1 << 20, 1 << 20),
        rec(0x9000, RegionKind::Private, 1 << 20, 1 << 20),
        rec(0x5000, RegionKind::Private, 2 << 20, 2 << 20),
    ])
}

/// Regions sharing an allocation base fold into one allocation, even when the
/// walk reports them apart.
#[test]
fn regions_fold_into_their_allocation() {
    let p = sample();
    assert_eq!(p.allocations.len(), 4);
    let heap = p
        .allocations
        .iter()
        .find(|a| a.alloc_base == 0x9000)
        .unwrap();
    assert_eq!(heap.committed, (1 << 30) + (1 << 20));
    assert_eq!(heap.resident, 4 << 20);
    let image = p
        .allocations
        .iter()
        .find(|a| a.alloc_base == 0x1000)
        .unwrap();
    assert_eq!((image.committed, image.resident), (12_288, 4096));
}

#[test]
fn totals_are_per_kind() {
    let p = sample();
    assert_eq!(
        p.total(RegionKind::Private),
        KindTotal {
            committed: (1 << 30) + (3 << 20),
            resident: 6 << 20,
        }
    );
    assert_eq!(
        p.total(RegionKind::Mapped),
        KindTotal {
            committed: 1 << 20,
            resident: 1 << 20,
        }
    );
}

/// Largest resident first; a tie goes to the lower base.
#[test]
fn largest_resident_orders_by_resident_bytes() {
    let p = ProcessRam::from_records([
        rec(0x3000, RegionKind::Private, 10, 5),
        rec(0x1000, RegionKind::Private, 10, 9),
        rec(0x2000, RegionKind::Private, 10, 5),
    ]);
    let bases: Vec<u64> = p.largest_resident(2).iter().map(|a| a.alloc_base).collect();
    assert_eq!(bases, vec![0x1000, 0x2000]);
}

/// Private allocations bucket by committed size, rounded up to a power of two;
/// images and views are left out.
#[test]
fn private_allocations_bucket_by_size() {
    let p = ProcessRam::from_records([
        rec(0x1000, RegionKind::Private, 16 << 20, 14 << 20),
        rec(0x2000, RegionKind::Private, 12 << 20, 12 << 20),
        rec(0x3000, RegionKind::Private, 4096, 4096),
        rec(0x4000, RegionKind::Image, 16 << 20, 16 << 20),
    ]);
    assert_eq!(
        p.private_by_size(),
        vec![
            SizeBucket {
                upper: 4096,
                count: 1,
                committed: 4096,
                resident: 4096,
            },
            SizeBucket {
                upper: 16 << 20,
                count: 2,
                committed: 28 << 20,
                resident: 26 << 20,
            },
        ]
    );
}

/// The walk runs on this platform and finds at least this test's own image.
#[cfg(windows)]
#[test]
fn capture_sees_the_running_image() {
    let p = ProcessRam::capture().expect("the walk runs on Windows");
    let image = p.total(RegionKind::Image);
    assert!(image.committed > 0 && image.resident > 0, "{image:?}");
    assert!(p
        .allocations
        .iter()
        .any(|a| a.kind == RegionKind::Image && a.name.is_some()));
}
