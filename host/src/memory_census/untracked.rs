//! Untracked memory (`docs/memory_census.md` §6.1): the commit charge none of
//! the census explains.
//!
//! `private − heap committed − pinned host memory − the VRAM backing`. The last
//! term is Windows': it commits system memory behind every device allocation,
//! untouched unless VRAM is evicted, about the size of the VRAM in use
//! (`evidence/motherd-oom-2026-10-24/FINDINGS.md`). Elsewhere the device's
//! memory is not in the process's commit, and the term is zero.

/// What the subtraction is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parts {
    pub private_bytes: u64,
    pub heap_committed_bytes: u64,
    pub pinned_bytes: u64,
    pub device_in_use_bytes: u64,
}

/// Whether this platform commits system memory behind VRAM.
pub const VRAM_IS_COMMITTED: bool = cfg!(windows);

/// The untracked remainder, clamped at zero.
pub fn untracked(p: &Parts) -> u64 {
    let backing = if VRAM_IS_COMMITTED {
        p.device_in_use_bytes
    } else {
        0
    };
    p.private_bytes
        .saturating_sub(p.heap_committed_bytes)
        .saturating_sub(p.pinned_bytes)
        .saturating_sub(backing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **BlackWorld at 20:20Z**: 225.70 GB private, 66.08 GB pinned, 74.73 GB
    /// of VRAM in use, and a heap committing 20 GB leave 64.89 GB untracked on
    /// Windows.
    #[test]
    fn untracked_is_what_the_parts_leave() {
        let p = Parts {
            private_bytes: 225_695_571_968,
            heap_committed_bytes: 20_000_000_000,
            pinned_bytes: 66_081_488_896,
            device_in_use_bytes: 74_732_154_880,
        };
        let expected = if VRAM_IS_COMMITTED {
            64_881_928_192
        } else {
            139_614_083_072
        };
        assert_eq!(untracked(&p), expected);
    }

    #[test]
    fn untracked_never_reads_below_zero() {
        let p = Parts {
            private_bytes: 1,
            heap_committed_bytes: 2,
            pinned_bytes: 3,
            device_in_use_bytes: 4,
        };
        assert_eq!(untracked(&p), 0);
    }
}
