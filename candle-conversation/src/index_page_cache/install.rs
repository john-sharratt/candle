//! Taking the cache's slab once, at startup, after the expert tier has pinned
//! its pool, and the process-wide handle every reader finds it by.
//!
//! The slab is taken by the first rung of the ladder that succeeds: the whole
//! target pinned; half pinned and half pageable; the whole target pageable;
//! then the pageable target halved down to [`INDEX_CACHE_MIN`]. A machine that
//! gets none of them, or whose target is zero, holds no cache, and every page
//! is read from the log.

use std::sync::OnceLock;

use candle::host_region::HostRegion;
use candle::vram::{PinnedUse, INDEX_CACHE_MIN};
use candle::Device;

use super::cache::IndexPageCache;

static CACHE: OnceLock<IndexPageCache> = OnceLock::new();

/// The process's cache, once [`install`] has taken it.
pub fn global() -> Option<&'static IndexPageCache> {
    CACHE.get()
}

/// Take the cache's slab for a `target` of bytes — once per process; later
/// calls answer the cache the first took. `device` is the one the pages are
/// uploaded to: pinning needs the calling thread bound to its context.
pub fn install(target: u64, device: &Device) -> Option<&'static IndexPageCache> {
    if let Some(cache) = CACHE.get() {
        return Some(cache);
    }
    if let Device::Cuda(d) = device {
        let _ = d.cuda_context().bind_to_thread();
    }
    let regions = ladder(
        target as usize,
        |n| HostRegion::pinned(n, PinnedUse::IndexPages).ok(),
        |n| HostRegion::pageable(n).ok(),
    );
    let Some(regions) = regions else {
        tracing::warn!(
            target_bytes = target,
            "index page cache: no slab could be taken; every index page is read from the log"
        );
        return None;
    };
    let cache = IndexPageCache::new(regions);
    let stats = cache.stats();
    tracing::info!(
        target_bytes = target,
        pinned_bytes = stats.pinned_bytes,
        pageable_bytes = stats.pageable_bytes,
        "index page cache: slab taken"
    );
    let _ = CACHE.set(cache);
    CACHE.get()
}

/// A cache over `regions` as the process's, for a test that reads through it;
/// the first install in a process is kept, as [`install`] keeps it.
#[cfg(test)]
pub(crate) fn install_for_tests(
    regions: impl FnOnce() -> Vec<HostRegion>,
) -> &'static IndexPageCache {
    CACHE.get_or_init(|| IndexPageCache::new(regions()))
}

/// The regions the first rung that succeeds takes, `pin` and `page` each
/// answering a region of the bytes asked for or `None`.
fn ladder<R>(
    target: usize,
    mut pin: impl FnMut(usize) -> Option<R>,
    mut page: impl FnMut(usize) -> Option<R>,
) -> Option<Vec<R>> {
    if target == 0 {
        return None;
    }
    if let Some(r) = pin(target) {
        return Some(vec![r]);
    }
    let half = target / 2;
    if let Some(p) = pin(half) {
        if let Some(q) = page(target - half) {
            return Some(vec![p, q]);
        }
    }
    let mut want = target;
    while want as u64 >= INDEX_CACHE_MIN {
        if let Some(q) = page(want) {
            return Some(vec![q]);
        }
        want /= 2;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;

    /// What each rung of the ladder was offered, and what it took, for an
    /// allocator that grants pinned bytes up to `pin_max` and pageable bytes
    /// up to `page_max`.
    fn climb(target: usize, pin_max: usize, page_max: usize) -> Option<Vec<(char, usize)>> {
        ladder(
            target,
            |n| (n <= pin_max).then_some(('p', n)),
            |n| (n <= page_max).then_some(('q', n)),
        )
    }

    /// The whole target pinned when the driver grants it.
    #[test]
    fn the_whole_target_is_pinned_first() {
        assert_eq!(climb(8 * GIB, usize::MAX, 0), Some(vec![('p', 8 * GIB)]));
    }

    /// Short of that, half pinned and half pageable.
    #[test]
    fn then_half_pinned_and_half_pageable() {
        assert_eq!(
            climb(8 * GIB, 4 * GIB, usize::MAX),
            Some(vec![('p', 4 * GIB), ('q', 4 * GIB)])
        );
    }

    /// With nothing pinned, the whole target pageable, then halved down to
    /// the 1 GiB floor — and nothing below it.
    #[test]
    fn then_pageable_halving_to_the_floor() {
        assert_eq!(climb(8 * GIB, 0, usize::MAX), Some(vec![('q', 8 * GIB)]));
        assert_eq!(climb(8 * GIB, 0, 3 * GIB), Some(vec![('q', 2 * GIB)]));
        assert_eq!(climb(8 * GIB, 0, GIB), Some(vec![('q', GIB)]));
        assert_eq!(climb(8 * GIB, 0, GIB - 1), None);
    }

    /// Half pinned with no pageable half to go beside it falls through to
    /// the pageable rungs rather than keeping half a slab.
    #[test]
    fn a_pinned_half_without_its_pageable_half_is_not_kept() {
        assert_eq!(climb(8 * GIB, 4 * GIB, 2 * GIB), Some(vec![('q', 2 * GIB)]));
    }

    /// A target of zero takes nothing.
    #[test]
    fn no_target_no_slab() {
        assert_eq!(climb(0, usize::MAX, usize::MAX), None);
    }
}
