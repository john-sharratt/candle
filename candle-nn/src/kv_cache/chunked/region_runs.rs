//! Claimed regions coalesced into the stretches that really are contiguous.
//!
//! The KV side hands regions back in whatever order its free list holds them, so
//! a claim of several regions is a *set* of addresses, not a range. A tenant whose
//! buffers are larger than one region — a guest's tensors, a creep's residual —
//! can only lay a buffer across regions whose addresses are consecutive: between
//! two runs lies ground another tenant holds, and every address in the span is
//! mapped, so a write that strays across the gap succeeds and corrupts it. This
//! is the arithmetic that turns the set into runs. Pure, so every rule is tested
//! without a card.

/// A maximal stretch of claimed regions whose addresses are consecutive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionRun {
    pub base: u64,
    pub bytes: usize,
}

/// Coalesce claimed region bases into contiguous runs, widest first.
///
/// Sorted first, because the claim order is the free list's order and says
/// nothing about adjacency — two regions claimed one after another may sit at
/// either end of the span. Widest first, so the largest ask meets the largest
/// stretch whatever order a caller places in. A repeated base is counted once:
/// coalesced twice it would invent a run twice as long as the ground.
pub fn runs_from(mut bases: Vec<u64>, region_bytes: usize) -> Vec<RegionRun> {
    bases.sort_unstable();
    bases.dedup();
    let mut runs: Vec<RegionRun> = Vec::new();
    for base in bases {
        match runs.last_mut() {
            Some(r) if r.base + r.bytes as u64 == base => r.bytes += region_bytes,
            _ => runs.push(RegionRun {
                base,
                bytes: region_bytes,
            }),
        }
    }
    runs.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.base.cmp(&b.base)));
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two adjacent regions (0x4000, 0x5000) and one on its own (0x1000), handed
    /// back out of address order, make two runs — the joined one first.
    #[test]
    fn adjacent_regions_coalesce_into_one_stretch() {
        assert_eq!(
            runs_from(vec![0x4000, 0x1000, 0x5000], 0x1000),
            vec![
                RegionRun {
                    base: 0x4000,
                    bytes: 0x2000
                },
                RegionRun {
                    base: 0x1000,
                    bytes: 0x1000
                },
            ]
        );
    }

    #[test]
    fn the_widest_run_comes_first() {
        let runs = runs_from(vec![0x1000, 0x8000, 0x9000, 0xA000], 0x1000);
        assert_eq!(
            runs[0],
            RegionRun {
                base: 0x8000,
                bytes: 0x3000
            }
        );
    }

    #[test]
    fn a_repeated_base_does_not_invent_ground() {
        assert_eq!(
            runs_from(vec![0x4000, 0x4000], 0x1000),
            vec![RegionRun {
                base: 0x4000,
                bytes: 0x1000
            }]
        );
    }

    #[test]
    fn no_regions_make_no_runs() {
        assert!(runs_from(Vec::new(), 0x1000).is_empty());
    }
}
