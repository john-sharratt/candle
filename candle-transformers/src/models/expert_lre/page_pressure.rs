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
///
/// The write is what matters: a zeroed allocation is committed lazily and
/// moves no pages, while filling it makes every page resident. `black_box`
/// keeps the fill from being elided as a dead store.
pub(crate) fn press(bytes: usize) {
    let buf = vec![1u8; bytes];
    std::hint::black_box(&buf);
}
