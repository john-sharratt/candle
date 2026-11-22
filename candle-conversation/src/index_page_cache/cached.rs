//! A cached page's bytes: a range of one slab block, readable for as long as
//! any holder keeps it.
//!
//! The block is not handed back to the slab while any `CachedBytes` for it
//! lives: the cache evicts an entry only when its own reference is the last
//! one. The slab's regions are kept alive by every `CachedBytes`, so the bytes
//! stay valid even past the cache itself.

use std::sync::Arc;

use candle::host_region::HostRegion;

/// See the module documentation.
pub struct CachedBytes {
    _regions: Arc<Vec<HostRegion>>,
    ptr: *const u8,
    len: usize,
}

// SAFETY: the bytes are written once, by the reader that filled the block,
// before this value exists, and only read after; the memory is kept alive by
// `_regions` and not re-tenanted while this value lives.
unsafe impl Send for CachedBytes {}
// SAFETY: as for `Send` — every access is a read.
unsafe impl Sync for CachedBytes {}

impl CachedBytes {
    /// # Safety
    /// `ptr..ptr + len` lies inside one of `regions`, holds the page's bytes,
    /// and is not written again while the value lives.
    pub(super) unsafe fn new(regions: Arc<Vec<HostRegion>>, ptr: *const u8, len: usize) -> Self {
        Self {
            _regions: regions,
            ptr,
            len,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: see `new`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::fmt::Debug for CachedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedBytes")
            .field("len", &self.len)
            .finish()
    }
}
