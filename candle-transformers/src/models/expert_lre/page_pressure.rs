//! Make the OS give physical RAM back before a page-lock retry.
//!
//! `cuMemAllocHost` needs resident, lockable pages. Free RAM is the one thing
//! it will not create: when other processes' idle working sets and the file
//! cache hold the pages, a large request is refused even though the machine
//! could hand them over. Writing through a large pageable allocation forces the
//! memory manager to do exactly that — trim other working sets, write their
//! dirty pages to the pagefile, drop standby cache — and freeing it leaves
//! those pages on the free list for the retry.

/// How much pageable memory one round of pressure writes through: twice the
/// warm tier's growth step, so a round frees at least the step it is paying for
/// after whatever the OS hands straight back to other processes.
pub(crate) const PAGE_PRESSURE_BYTES: usize = 1024 * 1024 * 1024;

/// Allocate `bytes` of ordinary pageable memory, write every byte, and free it.
/// Returns whether the round ran.
///
/// The write is what matters: a zeroed allocation is committed lazily and
/// moves no pages, while filling it makes every page resident. `black_box`
/// keeps the fill from being elided as a dead store.
///
/// **A refused allocation is an answer, not a failure.** This runs while the
/// warm pool grows, which is exactly when memory is short, and an infallible
/// allocation aborts the process on refusal — a model load dying where it
/// should have kept the pool it already held. A machine that cannot commit a
/// pageable GiB will not page-lock another one either, so `false` tells the
/// caller to stop growing.
pub(crate) fn press(bytes: usize) -> bool {
    let mut buf: Vec<u8> = Vec::new();
    if buf.try_reserve_exact(bytes).is_err() {
        return false;
    }
    buf.resize(bytes, 1);
    std::hint::black_box(&buf);
    true
}

#[cfg(test)]
mod tests {
    use super::press;

    /// A round the allocator cannot serve is reported, not aborted on.
    #[test]
    fn an_unservable_round_is_refused_rather_than_aborting() {
        assert!(!press(usize::MAX));
    }

    #[test]
    fn a_small_round_runs() {
        assert!(press(1 << 20));
    }
}
