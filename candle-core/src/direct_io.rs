//! Direct-I/O reader: positioned reads that bypass the OS page cache.
//!
//! Two subsystems read large, sector-aligned records off NVMe straight
//! into host memory the GPU will DMA from: the substrate's cold-load
//! path (`candle-conversation`, chunk records) and the expert cache's
//! cold tier (`candle-transformers`, one repacked expert per record).
//! Both want the same thing and neither can get it from
//! `seek + read_exact`, which goes through the page cache — on Windows
//! that caps NVMe sequential reads at ~7–10 MB/s when the destination is
//! `cuMemHostAlloc`'d, and even with a pageable landing pad it still pays
//! the cache-fill round trip.
//!
//! This module is **direct I/O** — the NVMe controller DMAs straight into
//! the caller's aligned host buffer — plus **per-stripe scoped threads**
//! to keep multiple NVMe submissions in flight at once. The disk-bound
//! step then runs at the device's actual sequential bandwidth (multi-GB/s
//! on a Gen4/Gen5 SSD).
//!
//! It lives in `candle-core` because it is an OS primitive with no
//! knowledge of what the bytes mean, and because the expert cache sits
//! below `candle-conversation` in the crate graph and so could not reach
//! it where it was first written.
//!
//! ## Alignment requirements
//!
//! Direct I/O imposes three sector-alignment constraints:
//!
//! - file offsets must be multiples of the volume sector size,
//! - read lengths must be multiples of the sector size,
//! - buffer addresses must be sector-aligned.
//!
//! We hardcode the alignment to **4 KiB** ([`DIRECT_IO_SECTOR`]). Modern
//! NVMe drives expose a 4 KiB logical sector; older 512 e drives
//! over-align to 4 KiB harmlessly. Both callers pad their records to
//! 4 KiB on disk, so stripe offsets and stripe lengths fall on sector
//! boundaries naturally. Buffer addresses are aligned by
//! [`AlignedScratch`] below, or by `cuMemAllocHost`, which returns
//! page-aligned memory.
//!
//! ## Cross-platform
//!
//! - **Linux**: `O_DIRECT` on open, `pread64` for positioned reads (via
//!   `std::os::unix::fs::FileExt::read_exact_at`, which is thread-safe).
//! - **Windows**: `FILE_FLAG_NO_BUFFERING` on open, `ReadFile` with the
//!   per-call `OVERLAPPED.Offset/OffsetHigh` (thread-safe positioned
//!   read; the handle is *not* opened with `FILE_FLAG_OVERLAPPED`, so
//!   the call is synchronous but the offset is per-call rather than
//!   shared file-pointer state).

use std::fs::File;
use std::io;
use std::path::Path;

/// Sector size we align every direct-I/O read to. 4 KiB covers every
/// modern NVMe sector size (512 e, 4 Kn, 4 KiB logical) and matches the
/// on-disk record padding both callers use.
pub const DIRECT_IO_SECTOR: usize = 4096;

/// Maximum NVMe queue depth a single [`DirectFile`] will keep in flight.
///
/// Sized for the production-target workstation: 4-drive Gen5 NVMe RAID 0
/// (~64 GB/s aggregate), where the saturating regime is QD16 — each of
/// the 4 drives runs at QD≈4 internally (assuming a 256 KiB OS RAID
/// stripe size, every 1 MiB application read spans 4 drives). 16 is
/// also comfortable on Gen4 (single drive: QD8–16 hits peak) and on
/// the dev-box Mobile Gen4 (~3 GB/s, QD8 saturates, QD16 has no
/// downside).
///
/// On Windows, getting actual QD16 at the device requires **N
/// independent file handles**, not just N threads sharing one handle —
/// `ReadFile` on a handle opened *without* `FILE_FLAG_OVERLAPPED` is
/// synchronous, and the storage stack funnels concurrent reads on a
/// single handle through a serialised IRP queue. So [`DirectFile`]
/// opens this many handles up front and each worker thread reads
/// through its own. Per-handle cost on open is ~5–50 µs; one-time at
/// open.
pub const MAX_CONCURRENT_READS: usize = 16;

/// A sector-aligned host scratch buffer — the destination for direct-I/O
/// reads. Grown on demand; reused across loads.
///
/// Backed by `std::alloc::alloc_zeroed` with a 4 KiB layout so the
/// buffer's base address satisfies the direct-I/O alignment requirement.
/// `Vec<u8>` uses the system allocator's natural alignment (8 or 16
/// bytes), which is *not* sector-aligned — Windows `ReadFile` with
/// `FILE_FLAG_NO_BUFFERING` rejects misaligned destinations with
/// `ERROR_INVALID_PARAMETER`.
pub struct AlignedScratch {
    ptr: *mut u8,
    capacity: usize,
}

unsafe impl Send for AlignedScratch {}
unsafe impl Sync for AlignedScratch {}

impl AlignedScratch {
    /// An empty scratch — no allocation yet. The first `ensure` call
    /// performs the aligned allocation.
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            capacity: 0,
        }
    }

    /// Bytes currently allocated.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Grow to at least `len` bytes, rounded up to the next sector
    /// boundary. The buffer's base address remains 4 KiB-aligned.
    pub fn ensure(&mut self, len: usize) -> io::Result<()> {
        let need = round_up_sector(len);
        if need <= self.capacity {
            return Ok(());
        }
        let layout = std::alloc::Layout::from_size_align(need, DIRECT_IO_SECTOR).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("aligned layout: {e}"))
        })?;
        let new_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if new_ptr.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("AlignedScratch alloc of {need} bytes failed"),
            ));
        }
        if !self.ptr.is_null() {
            let old_layout =
                std::alloc::Layout::from_size_align(self.capacity, DIRECT_IO_SECTOR).unwrap();
            unsafe { std::alloc::dealloc(self.ptr, old_layout) };
        }
        self.ptr = new_ptr;
        self.capacity = need;
        Ok(())
    }

    /// Borrow the first `len` bytes as a mutable slice. Caller asserts
    /// `len <= capacity`.
    pub fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        debug_assert!(len <= self.capacity);
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }

    /// Borrow the first `len` bytes as a slice. Caller asserts
    /// `len <= capacity`.
    pub fn as_slice(&self, len: usize) -> &[u8] {
        debug_assert!(len <= self.capacity);
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }
}

impl Default for AlignedScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AlignedScratch {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.capacity > 0 {
            let layout =
                std::alloc::Layout::from_size_align(self.capacity, DIRECT_IO_SECTOR).unwrap();
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }
}

/// Round `n` up to the next [`DIRECT_IO_SECTOR`] boundary.
///
/// Public because a caller laying out records in a file has to pad each
/// one to the same boundary the reader will demand of it.
pub fn round_up_sector(n: usize) -> usize {
    (n + DIRECT_IO_SECTOR - 1) & !(DIRECT_IO_SECTOR - 1)
}

/// One stripe to read: the file offset and a sector-aligned, non-overlapping
/// destination slice. Built by the caller from its own batch.
pub struct StripeRead<'a> {
    pub file_offset: u64,
    pub dest: &'a mut [u8],
}

/// A direct-I/O read handle on a file — opened with O_DIRECT
/// (Linux) / FILE_FLAG_NO_BUFFERING (Windows). Independent of any
/// buffered-write handle on the same path; reads through this handle
/// bypass the page cache.
///
/// Holds [`MAX_CONCURRENT_READS`] independent file descriptors on the
/// same path so the concurrent read pass can dispatch one positioned
/// `ReadFile` / `pread` per handle. Per-handle dispatch is the only
/// way to reach true QD>1 on Windows when the handle is opened in
/// synchronous mode (no `FILE_FLAG_OVERLAPPED`); a single handle
/// serialises concurrent reads through one IRP queue regardless of
/// how many threads call `ReadFile` on it.
///
/// `Send + Sync`: positioned reads via `pread64` /
/// `ReadFile`+`OVERLAPPED` are thread-safe, and the inner `Vec<File>`
/// is read-only after construction.
pub struct DirectFile {
    handles: Vec<File>,
}

impl DirectFile {
    /// Open `path` [`MAX_CONCURRENT_READS`] times with the platform's
    /// direct-I/O flag set. All handles refer to the same underlying
    /// file but maintain independent kernel-side I/O queues.
    pub fn open(path: &Path) -> io::Result<DirectFile> {
        let mut handles = Vec::with_capacity(MAX_CONCURRENT_READS);
        for _ in 0..MAX_CONCURRENT_READS {
            handles.push(open_direct(path)?);
        }
        Ok(DirectFile { handles })
    }

    /// Read `dest.len()` bytes at `offset`. Caller is responsible for
    /// sector-alignment of all three: `offset`, `dest.len()`, and the
    /// pointer `dest.as_mut_ptr()`. Used for the single-stripe case;
    /// concurrent batches go through [`Self::read_stripes_concurrent`].
    pub fn read_at(&self, offset: u64, dest: &mut [u8]) -> io::Result<()> {
        debug_alignment(offset, dest);
        pread_exact(&self.handles[0], offset, dest)
    }

    /// Read `dest.len()` bytes at `offset` using a specific handle
    /// from the pool. Reader workers each own a fixed handle index so
    /// concurrent reads dispatch to independent kernel I/O queues (true
    /// QD>1 on Windows).
    ///
    /// `handle_idx` is taken modulo [`MAX_CONCURRENT_READS`] so
    /// callers can hash arbitrary worker IDs through.
    pub fn read_at_with_handle(
        &self,
        handle_idx: usize,
        offset: u64,
        dest: &mut [u8],
    ) -> io::Result<()> {
        debug_alignment(offset, dest);
        let file = &self.handles[handle_idx % self.handles.len()];
        pread_exact(file, offset, dest)
    }

    /// Number of independent file handles the pool owns — also the
    /// natural reader-thread count for a pipelined load.
    pub fn n_handles(&self) -> usize {
        self.handles.len()
    }

    /// Submit every stripe in `stripes` across the
    /// [`MAX_CONCURRENT_READS`] handles, then join. Returns the first
    /// error if any read failed, otherwise `Ok(())`.
    ///
    /// Concurrency model: bucket the stripes into at most
    /// [`MAX_CONCURRENT_READS`] contiguous groups and spawn one
    /// scoped thread per non-empty group, each driving a single
    /// handle to drain its bucket sequentially. The kernel sees one
    /// outstanding NVMe submission per handle ⇒ true QD = number of
    /// non-empty buckets, regardless of platform.
    ///
    /// For ≤ [`MAX_CONCURRENT_READS`] stripes that's QD = stripes.len()
    /// with one stripe per worker. For larger batches each worker
    /// drains its 2–N stripes serially through its handle, but the
    /// queue stays full at the device — the next stripe submits as
    /// soon as the previous one's completion lands.
    pub fn read_stripes_concurrent(&self, stripes: &mut [StripeRead<'_>]) -> io::Result<()> {
        if stripes.is_empty() {
            return Ok(());
        }
        if stripes.len() == 1 {
            return self.read_at(stripes[0].file_offset, stripes[0].dest);
        }

        let n_workers = stripes.len().min(self.handles.len());
        let chunk_size = stripes.len().div_ceil(n_workers);

        std::thread::scope(|s| {
            let mut join_handles = Vec::with_capacity(n_workers);
            for (worker_idx, bucket) in stripes.chunks_mut(chunk_size).enumerate() {
                let file = &self.handles[worker_idx];
                join_handles.push(s.spawn(move || -> io::Result<()> {
                    for stripe in bucket.iter_mut() {
                        pread_exact(file, stripe.file_offset, stripe.dest)?;
                    }
                    Ok(())
                }));
            }
            let mut first_err: Option<io::Error> = None;
            for h in join_handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                    Err(_) => {
                        if first_err.is_none() {
                            first_err = Some(io::Error::other("DirectFile worker thread panicked"));
                        }
                    }
                }
            }
            match first_err {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })
    }
}

#[inline]
fn debug_alignment(offset: u64, dest: &[u8]) {
    debug_assert_eq!(
        offset as usize % DIRECT_IO_SECTOR,
        0,
        "direct-I/O offset {offset} not sector-aligned ({DIRECT_IO_SECTOR})",
    );
    debug_assert_eq!(
        dest.len() % DIRECT_IO_SECTOR,
        0,
        "direct-I/O length {} not sector-aligned ({DIRECT_IO_SECTOR})",
        dest.len(),
    );
    debug_assert_eq!(
        dest.as_ptr() as usize % DIRECT_IO_SECTOR,
        0,
        "direct-I/O dest pointer not sector-aligned ({DIRECT_IO_SECTOR})",
    );
}

// ── Platform-specific open + pread implementations ──────────────────────

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

/// Unix without `O_DIRECT` — macOS and the BSDs, where the flag does not exist
/// (`libc` does not define it for Apple targets at all, so naming it here is a
/// build failure rather than a slow path).
///
/// These platforms open buffered. Reads are correct and the alignment
/// invariants still hold; what is lost is the cache bypass, so a large read
/// also populates the page cache it was meant to avoid. Acceptable because
/// neither is a deployment target for this engine — Windows is the dev machine
/// and Linux the production one — and a build that does not compile on a
/// contributor's laptop is worse than one that reads through the cache on it.
///
/// macOS could get most of the way there with `fcntl(F_NOCACHE)`; it is not
/// written here because nothing in reach can test it.
#[cfg(all(unix, not(target_os = "linux")))]
fn open_direct(path: &Path) -> io::Result<File> {
    std::fs::OpenOptions::new().read(true).open(path)
}

#[cfg(unix)]
fn pread_exact(file: &File, offset: u64, dest: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dest, offset)
}

#[cfg(windows)]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_NO_BUFFERING;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open(path)
}

#[cfg(windows)]
fn pread_exact(file: &File, offset: u64, dest: &mut [u8]) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::TRUE;
    use windows_sys::Win32::Storage::FileSystem::ReadFile;
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let handle = file.as_raw_handle();
    let mut total_read: usize = 0;
    while total_read < dest.len() {
        let chunk = &mut dest[total_read..];
        // Cap a single ReadFile at u32::MAX (Windows API limit).
        let to_read = chunk.len().min(u32::MAX as usize) as u32;
        let pos = offset + total_read as u64;

        // OVERLAPPED carries the per-call file offset. Since the handle
        // is *not* FILE_FLAG_OVERLAPPED, ReadFile blocks until the I/O
        // completes; the OVERLAPPED struct is only used for its
        // offset fields (thread-safe positioned read).
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        // SAFETY: writing to the anonymous union's Offset/OffsetHigh
        // fields is the documented way to set the per-call offset.
        unsafe {
            let anon = &mut ov.Anonymous.Anonymous;
            anon.Offset = pos as u32;
            anon.OffsetHigh = (pos >> 32) as u32;
        }

        let mut bytes_read: u32 = 0;
        let ok = unsafe {
            ReadFile(
                handle as _,
                chunk.as_mut_ptr(),
                to_read,
                &mut bytes_read,
                &mut ov,
            )
        };
        if ok != TRUE {
            return Err(io::Error::last_os_error());
        }
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("direct ReadFile returned 0 bytes at offset {pos} (asked {to_read})"),
            ));
        }
        total_read += bytes_read as usize;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("candle_direct_io_{tag}_{nanos}.bin"));
        p
    }

    /// `AlignedScratch` allocates 4 KiB-aligned memory and rounds the
    /// requested length up to the next sector.
    #[test]
    fn aligned_scratch_is_sector_aligned() {
        let mut s = AlignedScratch::new();
        assert_eq!(s.capacity(), 0);
        s.ensure(100).unwrap();
        assert_eq!(s.capacity() % DIRECT_IO_SECTOR, 0);
        assert!(s.capacity() >= 100);
        let ptr = s.as_mut_slice(100).as_ptr();
        assert_eq!(ptr as usize % DIRECT_IO_SECTOR, 0);
    }

    /// Growing the scratch past existing capacity does not change the
    /// sector alignment invariant.
    #[test]
    fn aligned_scratch_grows_aligned() {
        let mut s = AlignedScratch::new();
        s.ensure(4096).unwrap();
        let _ = s.as_mut_slice(4096);
        s.ensure(8 * 4096).unwrap();
        let ptr = s.as_mut_slice(8 * 4096).as_ptr();
        assert_eq!(ptr as usize % DIRECT_IO_SECTOR, 0);
        assert!(s.capacity() >= 8 * 4096);
    }

    /// `round_up_sector` is the padding rule a writer must follow for the
    /// reader's alignment demands to hold.
    #[test]
    fn round_up_sector_pads_to_the_next_boundary() {
        assert_eq!(round_up_sector(0), 0);
        assert_eq!(round_up_sector(1), DIRECT_IO_SECTOR);
        assert_eq!(round_up_sector(DIRECT_IO_SECTOR), DIRECT_IO_SECTOR);
        assert_eq!(round_up_sector(DIRECT_IO_SECTOR + 1), 2 * DIRECT_IO_SECTOR);
    }

    /// End-to-end direct-I/O read: write a sector-aligned file
    /// buffered, then read it back via `DirectFile` and verify byte
    /// equality.
    ///
    /// The reader is `O_DIRECT`/`FILE_FLAG_NO_BUFFERING`; the bytes we
    /// just wrote are still in the kernel cache, but we use the
    /// **direct** path explicitly so the read goes through the cache
    /// bypass — this also exercises the alignment-check `debug_assert`s.
    #[test]
    fn direct_file_roundtrip() {
        let path = tmp_path("roundtrip");
        let mut payload = vec![0u8; 4 * DIRECT_IO_SECTOR];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u32 * 17 + 3) as u8;
        }
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
            f.sync_all().unwrap();
        }
        let df = DirectFile::open(&path).unwrap();
        let mut scratch = AlignedScratch::new();
        scratch.ensure(payload.len()).unwrap();
        let dest = scratch.as_mut_slice(payload.len());
        df.read_at(0, dest).unwrap();
        assert_eq!(dest, payload.as_slice());
        drop(df);
        std::fs::remove_file(&path).ok();
    }

    /// Concurrent multi-stripe read: split a file into N non-overlapping
    /// sector-aligned stripes, submit them via
    /// `read_stripes_concurrent`, and verify each stripe matches the
    /// expected byte range. This is the cold-load shape (one read per
    /// disk-locality region, all in flight at once).
    #[test]
    fn direct_file_concurrent_stripes() {
        let path = tmp_path("concurrent");
        let n_stripes = 4;
        let stripe_bytes = 8 * DIRECT_IO_SECTOR;
        let mut payload = vec![0u8; n_stripes * stripe_bytes];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u32 * 31 + 7) as u8;
        }
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
            f.sync_all().unwrap();
        }
        let df = DirectFile::open(&path).unwrap();
        let mut scratch = AlignedScratch::new();
        scratch.ensure(payload.len()).unwrap();
        let dest = scratch.as_mut_slice(payload.len());
        // Carve `dest` into N non-overlapping sector-aligned chunks
        // and build the StripeRead descriptors over them.
        let mut chunks: Vec<&mut [u8]> = Vec::with_capacity(n_stripes);
        let mut rem = dest;
        for _ in 0..n_stripes {
            let (head, tail) = rem.split_at_mut(stripe_bytes);
            chunks.push(head);
            rem = tail;
        }
        let mut stripes: Vec<StripeRead<'_>> = chunks
            .into_iter()
            .enumerate()
            .map(|(i, dest)| StripeRead {
                file_offset: (i * stripe_bytes) as u64,
                dest,
            })
            .collect();
        df.read_stripes_concurrent(&mut stripes).unwrap();
        // Verify the whole buffer round-trips.
        let got = scratch.as_slice(payload.len());
        assert_eq!(got, payload.as_slice());
        drop(df);
        std::fs::remove_file(&path).ok();
    }
}
