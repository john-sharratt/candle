//! Where an index cache's device buffers live: slots of the span's QSA-index
//! arenas.
//!
//! An [`IndexCache`](super::indexer::IndexCache) holds three kinds of device
//! buffer, all of them carried from wave to wave and none of them the wave's:
//!
//! - **The live tail's keys**, as a table of fixed pages ([`KeyPages`]) of
//!   [`PAGE_BLOCKS`] block keys each. A page is one slot, so the tail grows a page
//!   at a time instead of reallocating and copying a buffer twice its size, and
//!   the whole tail is span ground its tenant's arenas account for.
//! - **The open block**, `MAX_RATIO` raw rows — one slot.
//! - **Rewind copies of the open block** ([`SnapshotBuffers`]), claimed with the
//!   cache so taking a snapshot inside a forward copies into a buffer that
//!   already stands instead of allocating.
//!
//! Every buffer is an anchored view of its slot, so no view of it outlives the
//! slot. On a device with no reservation — a CPU device, which is every host
//! test — the same buffers are ordinary tensors.
//!
//! The QSA-index arenas are packed between forwards like the recurrent-state ones
//! (`IndexCache::relocate`, driven by `compact_index_caches`): a buffer whose slot is
//! a planned source is copied onto its destination and rebuilt there.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{claim_arena_slots, relocate_tensor, ArenaSlot, SlotTenant};

use super::indexer::tensor_ptr;

/// Block keys per live-tail page: 128 KiB at the released 128-wide indexer
/// head, 128 pages to a region.
///
/// Wide enough that a deep tail is a few hundred descriptors in the scorer's page
/// table (256K tokens at ratio 4 is 256 pages), narrow enough that the unused end
/// of a sequence's last page is a rounding error beside its K/V.
pub const PAGE_BLOCKS: usize = 256;

/// Rewind copies of the open block one cache can have outstanding at once.
///
/// A wave takes one for its failure bracket, a speculative verify keeps the first
/// of those as the block's entering state for the rest of the step, and the draft
/// head takes one around its walk: three. One more is headroom, and running out is
/// refused by name rather than met with an allocation.
pub const SNAPSHOT_BUFFERS: usize = 4;

/// `n` uninitialised F32 buffers of `[rows, d]`: on CUDA, anchored views of
/// QSA-index arena slots claimed in one call; on any other device, ordinary
/// tensors.
///
/// **Between forwards.** A claim that needs a new arena takes the arena window,
/// which refuses inside a forward.
pub(super) fn alloc_buffers(
    device: &Device,
    rows: usize,
    d: usize,
    n: usize,
) -> Result<Vec<Tensor>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    match device {
        Device::Cuda(_) => claim_arena_slots(
            device,
            SlotTenant::QsaIndex,
            rows * d * DType::F32.size_in_bytes(),
            n,
        )?
        .into_iter()
        .map(|slot| Arc::new(slot).tensor(0, DType::F32, (rows, d), device))
        .collect(),
        _ => (0..n)
            .map(|_| Tensor::zeros((rows, d), DType::F32, device))
            .collect(),
    }
}

/// The live tail's block keys: page `p` holds blocks `[p·PAGE_BLOCKS,
/// (p+1)·PAGE_BLOCKS)`, row-major `[PAGE_BLOCKS, d]`.
#[derive(Debug)]
pub(super) struct KeyPages {
    pages: Vec<Tensor>,
    d: usize,
    device: Device,
}

impl KeyPages {
    pub(super) fn new(d: usize, device: &Device) -> Self {
        Self {
            pages: Vec::new(),
            d,
            device: device.clone(),
        }
    }

    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    pub(super) fn head_dim(&self) -> usize {
        self.d
    }

    /// Blocks the pages can hold.
    pub(super) fn capacity_blocks(&self) -> usize {
        self.pages.len() * PAGE_BLOCKS
    }

    /// Room for `blocks` block keys, claiming the missing pages in one call.
    /// Never shrinks, and never moves a key already written.
    pub(super) fn ensure(&mut self, blocks: usize) -> Result<()> {
        let want = blocks.div_ceil(PAGE_BLOCKS);
        if want > self.pages.len() {
            let more = alloc_buffers(&self.device, PAGE_BLOCKS, self.d, want - self.pages.len())?;
            self.pages.extend(more);
        }
        Ok(())
    }

    /// Every page's base address, in page order — what a caller resolving many
    /// rows at once reads once, then indexes with [`row_addr`].
    pub(super) fn page_ptrs(&self) -> Result<Vec<u64>> {
        self.pages.iter().map(tensor_ptr).collect()
    }

    /// The pages covering the first `n_blocks` keys, as `(base address, rows
    /// used)` — the live tail's descriptor-table entries.
    pub(super) fn tail(&self, n_blocks: usize) -> Result<Vec<(u64, usize)>> {
        self.check(n_blocks)?;
        (0..n_blocks.div_ceil(PAGE_BLOCKS))
            .map(|p| {
                let used = (n_blocks - p * PAGE_BLOCKS).min(PAGE_BLOCKS);
                Ok((tensor_ptr(&self.pages[p])?, used))
            })
            .collect()
    }

    /// The first `n_blocks` keys, `[n_blocks, d]` row-major, read back to the host
    /// page by page — for a seal or a record, never for the scorer, which reads the
    /// pages in place. Nothing is gathered on the device: a gather is a buffer from
    /// the CUDA pool, outside the span, for bytes that are only on their way off
    /// the card.
    pub(super) fn host_rows(&self, n_blocks: usize) -> Result<Vec<f32>> {
        self.check(n_blocks)?;
        let mut out = Vec::with_capacity(n_blocks * self.d);
        for p in 0..n_blocks.div_ceil(PAGE_BLOCKS) {
            let used = (n_blocks - p * PAGE_BLOCKS).min(PAGE_BLOCKS);
            out.extend(
                self.pages[p]
                    .narrow(0, 0, used)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
        Ok(out)
    }

    /// Write `rows` (`[n, d]` row-major, host) as the first `n` keys, claiming
    /// pages as needed — straight into the pages, with no device buffer between.
    pub(super) fn write_host_rows(&mut self, rows: &[f32]) -> Result<()> {
        if !rows.len().is_multiple_of(self.d) {
            candle::bail!(
                "qsa index pages: {} values is not a whole number of {}-wide rows",
                rows.len(),
                self.d
            );
        }
        let n = rows.len() / self.d;
        self.ensure(n)?;
        for p in 0..n.div_ceil(PAGE_BLOCKS) {
            let first = p * PAGE_BLOCKS;
            let used = (n - first).min(PAGE_BLOCKS);
            write_host(
                &self.pages[p],
                &rows[first * self.d..(first + used) * self.d],
            )?;
        }
        Ok(())
    }

    /// Hand over the pages holding the first `n_blocks` keys, leaving this table
    /// with the pages above them — how a live tail is closed into a page without
    /// moving a key.
    pub(super) fn detach(&mut self, n_blocks: usize) -> Result<Vec<Tensor>> {
        self.check(n_blocks)?;
        Ok(self.pages.drain(..n_blocks.div_ceil(PAGE_BLOCKS)).collect())
    }

    /// A copy of the first `n_blocks` keys in pages of their own — only the pages
    /// those keys reach, not this table's whole capacity.
    pub(super) fn fork(&self, n_blocks: usize) -> Result<Self> {
        self.check(n_blocks)?;
        let mut child = Self::new(self.d, &self.device);
        child.ensure(n_blocks)?;
        for p in 0..n_blocks.div_ceil(PAGE_BLOCKS) {
            let used = (n_blocks - p * PAGE_BLOCKS).min(PAGE_BLOCKS);
            child.pages[p].slice_set(&self.pages[p].narrow(0, 0, used)?, 0, 0)?;
        }
        Ok(child)
    }

    /// Give every page back — a sequence starting over holds no keys.
    pub(super) fn clear(&mut self) {
        self.pages.clear();
    }

    /// Move every page whose slot is a planned source — see [`relocate_tensor`].
    /// Answers how many moved. Every page moves whole, the dead rows above the
    /// live tail included: which rows are live is the cache's to know, not the
    /// page's, and a page is one copy either way.
    pub(super) fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        let mut moved = 0;
        for page in &mut self.pages {
            moved += usize::from(relocate_tensor(page, moves)?);
        }
        Ok(moved)
    }

    fn check(&self, n_blocks: usize) -> Result<()> {
        if n_blocks > self.capacity_blocks() {
            candle::bail!(
                "qsa index pages: {n_blocks} keys asked of {} page(s) holding {}",
                self.pages.len(),
                self.capacity_blocks()
            );
        }
        Ok(())
    }
}

/// Copy host `vals` into the front of `dst` — on CUDA one host-to-device copy into
/// the slot `dst` views, on the stream every reader of the slot uses; on any other
/// device a `slice_set`.
pub(super) fn write_host(dst: &Tensor, vals: &[f32]) -> Result<()> {
    if vals.len() > dst.elem_count() {
        candle::bail!(
            "qsa index: {} values into a {}-value buffer",
            vals.len(),
            dst.elem_count()
        );
    }
    match dst.device() {
        Device::Cuda(cuda) => {
            // `dst` is a live, dense F32 buffer of at least `vals.len()`
            // elements; the upload returns once `vals` is staged, so it may go.
            //
            // SAFETY: `vals` is plain f32s.
            let raw = unsafe {
                std::slice::from_raw_parts(vals.as_ptr() as *const u8, std::mem::size_of_val(vals))
            };
            cuda.upload_raw(tensor_ptr(dst)?, raw)
        }
        dev => {
            let (_, d) = dst.dims2()?;
            let rows = vals.len() / d;
            dst.slice_set(&Tensor::from_slice(vals, (rows, d), dev)?, 0, 0)
        }
    }
}

/// Device address of block key `k`, from [`KeyPages::page_ptrs`].
pub(super) fn row_addr(page_ptrs: &[u64], k: usize, d: usize) -> u64 {
    page_ptrs[k / PAGE_BLOCKS] + ((k % PAGE_BLOCKS) * d * DType::F32.size_in_bytes()) as u64
}

/// The free rewind copies of one cache's open block. Shared with every
/// [`SnapshotBuffer`] taken from it, each of which puts its buffer back on drop.
#[derive(Debug, Clone)]
pub(super) struct SnapshotBuffers(Arc<Mutex<Vec<Tensor>>>);

impl SnapshotBuffers {
    pub(super) fn new(buffers: Vec<Tensor>) -> Self {
        Self(Arc::new(Mutex::new(buffers)))
    }

    /// A free buffer. Refused by name when every one is out — that is more rewind
    /// points outstanding on one cache than any caller takes, which is a leak or a
    /// new caller, and either wants saying rather than an allocation mid-forward.
    pub(super) fn take(&self) -> Result<SnapshotBuffer> {
        let buf = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qsa index: all {SNAPSHOT_BUFFERS} rewind copies of this cache's open \
                     block are outstanding"
                ))
            })?;
        Ok(SnapshotBuffer {
            buf: Some(buf),
            home: self.clone(),
        })
    }

    /// Move every FREE buffer whose slot is a planned source. A buffer a snapshot
    /// holds is not reached and keeps its slot — the snapshot may be read at any
    /// moment until it drops — so its planned destination simply goes back.
    pub(super) fn relocate(&self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        let mut free = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut moved = 0;
        for buf in free.iter_mut() {
            moved += usize::from(relocate_tensor(buf, moves)?);
        }
        Ok(moved)
    }
}

/// One rewind copy of an open block, returned to its cache's
/// [`SnapshotBuffers`] on drop.
#[derive(Debug)]
pub(super) struct SnapshotBuffer {
    buf: Option<Tensor>,
    home: SnapshotBuffers,
}

impl SnapshotBuffer {
    pub(super) fn tensor(&self) -> &Tensor {
        self.buf.as_ref().expect("held until drop")
    }
}

impl Drop for SnapshotBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.home
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, d: usize) -> Vec<f32> {
        (0..n * d).map(|i| i as f32).collect()
    }

    #[test]
    fn a_row_address_walks_pages_of_page_blocks() {
        let ptrs = [0x1000u64, 0x9000];
        assert_eq!(row_addr(&ptrs, 0, 128), 0x1000);
        assert_eq!(row_addr(&ptrs, 1, 128), 0x1000 + 512);
        assert_eq!(row_addr(&ptrs, 255, 128), 0x1000 + 255 * 512);
        assert_eq!(row_addr(&ptrs, 256, 128), 0x9000);
        assert_eq!(row_addr(&ptrs, 257, 128), 0x9000 + 512);
    }

    #[test]
    fn pages_are_claimed_a_page_at_a_time() {
        let mut keys = KeyPages::new(4, &Device::Cpu);
        assert_eq!(keys.capacity_blocks(), 0);
        keys.ensure(1).unwrap();
        assert_eq!(keys.capacity_blocks(), PAGE_BLOCKS);
        keys.ensure(PAGE_BLOCKS).unwrap();
        assert_eq!(
            keys.capacity_blocks(),
            PAGE_BLOCKS,
            "one page covers it exactly"
        );
        keys.ensure(PAGE_BLOCKS + 1).unwrap();
        assert_eq!(keys.capacity_blocks(), 2 * PAGE_BLOCKS);
        keys.ensure(3).unwrap();
        assert_eq!(keys.capacity_blocks(), 2 * PAGE_BLOCKS, "never shrinks");
    }

    /// Rows written across a page boundary come back as the same rows, and the
    /// tail's descriptors name exactly the rows used in each page.
    #[test]
    fn rows_round_trip_across_a_page_boundary() {
        let n = PAGE_BLOCKS + 3;
        let src = rows(n, 4);
        let mut keys = KeyPages::new(4, &Device::Cpu);
        keys.write_host_rows(&src).unwrap();
        assert_eq!(keys.host_rows(n).unwrap(), src);
        let child = keys.fork(n).unwrap();
        assert_eq!(child.capacity_blocks(), 2 * PAGE_BLOCKS);
        assert_eq!(child.host_rows(n).unwrap(), src, "a fork carries the keys");
        assert!(
            keys.host_rows(2 * PAGE_BLOCKS + 1).is_err(),
            "past capacity is refused"
        );
    }

    /// A fork claims only the pages its keys reach, not the parent's capacity.
    #[test]
    fn a_fork_claims_only_the_pages_in_use() {
        let mut keys = KeyPages::new(4, &Device::Cpu);
        keys.ensure(4 * PAGE_BLOCKS).unwrap();
        keys.write_host_rows(&rows(10, 4)).unwrap();
        assert_eq!(keys.fork(10).unwrap().capacity_blocks(), PAGE_BLOCKS);
        assert_eq!(keys.fork(0).unwrap().capacity_blocks(), 0);
    }

    /// **Detaching hands the written pages over as they stand and keeps the rest.**
    /// The pages covering `n` keys leave with their rows; the table keeps its
    /// capacity above them, empty, for the tail that follows.
    #[test]
    fn detaching_hands_over_the_pages_in_use_and_keeps_the_rest() {
        let n = PAGE_BLOCKS + 3;
        let src = rows(n, 4);
        let mut keys = KeyPages::new(4, &Device::Cpu);
        keys.ensure(3 * PAGE_BLOCKS).unwrap();
        keys.write_host_rows(&src).unwrap();
        let taken = keys.detach(n).unwrap();
        assert_eq!(taken.len(), 2, "the two pages the keys reach");
        assert_eq!(keys.capacity_blocks(), PAGE_BLOCKS, "the unused page stays");
        let back: Vec<f32> = taken
            .iter()
            .flat_map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .take(n * 4)
            .collect();
        assert_eq!(back, src, "the rows leave where they were written");
    }

    #[test]
    fn snapshot_buffers_are_returned_on_drop_and_refused_when_all_are_out() {
        let pool = SnapshotBuffers::new(alloc_buffers(&Device::Cpu, 4, 4, 2).unwrap());
        let a = pool.take().unwrap();
        let b = pool.take().unwrap();
        assert!(pool.take().is_err(), "both buffers are out");
        drop(a);
        let c = pool.take().unwrap();
        assert_eq!(c.tensor().dims2().unwrap(), (4, 4));
        drop((b, c));
        assert_eq!(pool.0.lock().unwrap().len(), 2);
    }
}
