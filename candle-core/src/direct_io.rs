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
//! - **Windows**: `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` on open,
//!   `ReadFile` with the per-call `OVERLAPPED.Offset/OffsetHigh` and an
//!   event the caller then waits on. The caller blocks as before, but the
//!   request is queued at the drive beside every other thread's — on a
//!   synchronous handle the same reads completed one at a time (see
//!   `open_direct`).

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
/// [`DirectFile`] opens this many handles up front and each worker thread
/// reads through its own. On Windows the handles are overlapped, which is
/// what puts the reads in flight together at all (`open_direct`); one
/// handle per worker keeps their requests independent. Per-handle cost on
/// open is ~5–50 µs; one-time at open.
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
/// `ReadFile` / `pread` per handle. On Windows they are overlapped
/// handles, each read waiting on its own event — see `open_direct` for
/// the measurement that made them so.
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

/// Windows opens every handle **overlapped**. Measured on the 16 GB laptop's
/// SN810 (`tests::read_method_sweep`): random 1.9 MiB reads issued as blocking
/// `ReadFile`s — one thread and one synchronous handle each, eight at once —
/// completed one at a time, at 3.9 GB/s whatever the depth, while the same
/// reads as overlapped requests ran 6.3 GB/s at eight outstanding and 6.5 at
/// thirty-two. The synchronous path serialises them; an overlapped handle
/// queues them at the drive.
#[cfg(windows)]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED};
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
        .open(path)
}

/// A manual-reset event, closed on drop — the completion signal one
/// overlapped read waits on. Per call rather than per handle, because nothing
/// stops two threads reading through the same handle at once, and two reads
/// sharing an event would each take the other's completion as their own.
#[cfg(windows)]
struct ReadEvent(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl ReadEvent {
    fn new() -> io::Result<Self> {
        use windows_sys::Win32::System::Threading::CreateEventW;
        // SAFETY: no security attributes, manual reset, initially unsignalled,
        // unnamed — every argument is a plain value or null.
        let h = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        if h.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(h))
    }
}

#[cfg(windows)]
impl Drop for ReadEvent {
    fn drop(&mut self) {
        // SAFETY: the handle came from CreateEventW and is closed once.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

/// A positioned read on an overlapped handle, waited for: the request goes to
/// the drive's queue beside every other thread's, and this thread blocks on
/// its own completion only.
#[cfg(windows)]
fn pread_exact(file: &File, offset: u64, dest: &mut [u8]) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_IO_PENDING};
    use windows_sys::Win32::Storage::FileSystem::ReadFile;
    use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};

    let handle = file.as_raw_handle();
    let event = ReadEvent::new()?;
    let mut total_read: usize = 0;
    while total_read < dest.len() {
        let chunk = &mut dest[total_read..];
        // Cap a single ReadFile at u32::MAX (Windows API limit).
        let to_read = chunk.len().min(u32::MAX as usize) as u32;
        let pos = offset + total_read as u64;

        // OVERLAPPED carries the per-call file offset and the event the
        // completion signals.
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        // SAFETY: writing to the anonymous union's Offset/OffsetHigh
        // fields is the documented way to set the per-call offset.
        unsafe {
            let anon = &mut ov.Anonymous.Anonymous;
            anon.Offset = pos as u32;
            anon.OffsetHigh = (pos >> 32) as u32;
        }
        ov.hEvent = event.0;

        // SAFETY: `chunk` and `ov` outlive the request — GetOverlappedResult
        // below waits for it to complete before either is touched again.
        let ok = unsafe {
            ReadFile(
                handle as _,
                chunk.as_mut_ptr(),
                to_read,
                std::ptr::null_mut(),
                &mut ov,
            )
        };
        if ok == 0 {
            let e = unsafe { GetLastError() };
            if e != ERROR_IO_PENDING {
                return Err(io::Error::from_raw_os_error(e as i32));
            }
        }
        let mut bytes_read: u32 = 0;
        // SAFETY: waits for the request issued above on its own event.
        let ok = unsafe { GetOverlappedResult(handle as _, &ov, &mut bytes_read, 1) };
        if ok == 0 {
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
    use crate::quantized::gguf_file::Content;
    use std::io::{Seek, SeekFrom, Write};
    use std::path::PathBuf;

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

    /// Where a model pack's expert section sits: `(path, offset, len)`.
    type ExpertSection = (PathBuf, u64, u64);

    /// The expert section of `path`, when it is a model pack that has one —
    /// read from the pack's own `zen.pack.experts.*` metadata.
    fn expert_section(path: &Path) -> Option<ExpertSection> {
        let mut f = File::open(path).ok()?;
        let content = Content::read(&mut f).ok()?;
        let key = |k: &str| content.metadata.get(k)?.to_u64().ok();
        Some((
            path.to_path_buf(),
            key("zen.pack.experts.offset")?,
            key("zen.pack.experts.len")?,
        ))
    }

    /// Every expert section under the zend model cache, largest first — the
    /// multi-GB spans of the drive the expert cache reads in production. More
    /// than one, because a drive's read ceiling is a property of where a file's
    /// data landed as well as of the drive: a pack written onto a nearly full
    /// drive can read slower than an older one beside it.
    fn expert_packs() -> Vec<ExpertSection> {
        let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
        else {
            return Vec::new();
        };
        let mut stack = vec![Path::new(&home).join(".cache/zend/models")];
        let mut found = Vec::new();
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.to_string_lossy().ends_with(".pack.gguf") {
                    found.extend(expert_section(&p));
                }
            }
        }
        found.sort_by_key(|s| std::cmp::Reverse(s.2));
        found
    }

    /// **The drive's ceiling for the expert cache's read shape.** Random
    /// whole-record reads of `record` bytes at queue depths 1–16 (one thread
    /// and one handle per outstanding read, as the stager's readers), and the
    /// same record split into 4 concurrent quarter reads — which says whether a
    /// single expert lands faster striped than whole. Prints GB/s and the
    /// mean per-read latency; asserts only that every read succeeded.
    ///
    /// `cargo test --release -p candle-core --lib direct_io::tests::queue_depth_sweep -- --ignored --nocapture`
    #[test]
    #[ignore = "reads a multi-GB expert pack off the real drive for ~30 s"]
    fn queue_depth_sweep() {
        let packs = expert_packs();
        if packs.is_empty() {
            eprintln!("no model pack with an expert section under ~/.cache/zend/models");
        }
        for (path, base, len) in packs {
            sweep_pack(&path, base, len);
        }
    }

    /// [`queue_depth_sweep`] over one pack's expert section.
    fn sweep_pack(path: &Path, base: u64, len: u64) {
        use std::io::Read;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{Duration, Instant};
        let df = DirectFile::open(path).unwrap();
        // The section's own record stride (its header's bytes 24..32, already a
        // whole number of sectors), so the sweep reads the shape this rung's
        // experts actually have. The section starts sector-aligned, so every
        // `base + k × stride` is a legal direct-I/O offset.
        let mut head = [0u8; 32];
        let mut f = File::open(path).unwrap();
        f.seek(SeekFrom::Start(base)).unwrap();
        f.read_exact(&mut head).unwrap();
        assert_eq!(
            &head[..8],
            b"CNDLXPK2",
            "{} has no expert section",
            path.display()
        );
        let record = u64::from_le_bytes(head[24..32].try_into().unwrap()) as usize;
        assert_eq!(
            record,
            round_up_sector(record),
            "the stride is sector-aligned"
        );
        assert_eq!(
            base,
            round_up_sector(base as usize) as u64,
            "the section is sector-aligned"
        );
        let records = (len / record as u64).saturating_sub(1);
        if records == 0 {
            return;
        }
        eprintln!(
            "{} — {:.1} GiB section, {records} records of {record} B",
            path.display(),
            len as f64 / (1u64 << 30) as f64
        );
        let window = Duration::from_secs(3);
        // (threads, pieces per record): pieces > 1 splits each record into
        // that many concurrent reads, one thread each.
        for &(qd, pieces) in &[
            (1, 1),
            (2, 1),
            (4, 1),
            (8, 1),
            (16, 1),
            (1, 4),
            (2, 4),
            (4, 4),
        ] {
            let bytes = AtomicU64::new(0);
            let reads = AtomicU64::new(0);
            let busy_ns = AtomicU64::new(0);
            let piece = record / pieces;
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for w in 0..qd {
                    let (df, bytes, reads, busy_ns) = (&df, &bytes, &reads, &busy_ns);
                    s.spawn(move || {
                        let mut scratch = AlignedScratch::new();
                        scratch.ensure(record).unwrap();
                        // A cheap LCG: distinct random records per thread.
                        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (w as u64 + 1);
                        while t0.elapsed() < window {
                            x = x
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            let at = base + (x >> 16) % records * record as u64;
                            let t = Instant::now();
                            if pieces == 1 {
                                df.read_at_with_handle(w, at, scratch.as_mut_slice(record))
                                    .unwrap();
                            } else {
                                let dest = scratch.as_mut_slice(record);
                                std::thread::scope(|ps| {
                                    for (i, chunk) in dest.chunks_mut(piece).enumerate() {
                                        let off = at + (i * piece) as u64;
                                        let h = w * pieces + i;
                                        ps.spawn(move || {
                                            df.read_at_with_handle(h, off, chunk).unwrap()
                                        });
                                    }
                                });
                            }
                            busy_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            bytes.fetch_add(record as u64, Ordering::Relaxed);
                            reads.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
            let secs = t0.elapsed().as_secs_f64();
            let n = reads.load(Ordering::Relaxed).max(1);
            eprintln!(
                "  records in flight {qd:>2} × {pieces} piece(s): {:>6.2} GB/s, {:>6.2} ms per record, {n} records",
                bytes.load(Ordering::Relaxed) as f64 / secs / 1e9,
                busy_ns.load(Ordering::Relaxed) as f64 / n as f64 / 1e6,
            );
        }
        // **Read shape against placement**, at the stager's queue depth: reads
        // of `size` bytes at offsets that are multiples of `stride` from the
        // section base — the pack's own stride, then the same length on 64 KiB
        // and 2 MiB boundaries, then other lengths on their own boundaries. Says
        // whether the record's size or its alignment sets the ceiling.
        const KIB: usize = 1024;
        let span = len - record as u64;
        for &(size, stride) in &[
            (record, record),
            (record, 64 * KIB * record.div_ceil(64 * KIB)),
            (record, 2048 * KIB),
            (1024 * KIB, 1024 * KIB),
            (2048 * KIB, 2048 * KIB),
            (4096 * KIB, 4096 * KIB),
        ] {
            let slots = span / stride as u64;
            if slots == 0 {
                continue;
            }
            let bytes = AtomicU64::new(0);
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for w in 0..8usize {
                    let (df, bytes) = (&df, &bytes);
                    s.spawn(move || {
                        let mut scratch = AlignedScratch::new();
                        scratch.ensure(size).unwrap();
                        let mut x = 0xD1B5_4A32_D192_ED03u64 ^ (w as u64 + 1);
                        while t0.elapsed() < window {
                            x = x
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            let at = base + (x >> 16) % slots * stride as u64;
                            df.read_at_with_handle(w, at, scratch.as_mut_slice(size))
                                .unwrap();
                            bytes.fetch_add(size as u64, Ordering::Relaxed);
                        }
                    });
                }
            });
            eprintln!(
                "  shape {:>5} KiB every {:>5} KiB, 8 in flight: {:>6.2} GB/s",
                size / KIB,
                stride / KIB,
                bytes.load(Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64() / 1e9,
            );
        }
    }

    /// One configuration of [`read_method_sweep`]: overlapped reads of `size`
    /// bytes at random `size`-aligned offsets of each section in `files`, with
    /// `depth` outstanding per file spread over `handles` overlapped handles of
    /// that file, completed through one I/O completion port. Returns GB/s over
    /// every file together.
    #[cfg(windows)]
    fn overlapped_rate(
        files: &[(PathBuf, u64, u64)],
        size: usize,
        depth: usize,
        handles: usize,
        window: std::time::Duration,
    ) -> f64 {
        use std::os::windows::fs::OpenOptionsExt;
        use std::os::windows::io::AsRawHandle;
        use std::time::Instant;
        use windows_sys::Win32::Foundation::{
            CloseHandle, GetLastError, ERROR_IO_PENDING, INVALID_HANDLE_VALUE,
        };
        use windows_sys::Win32::Storage::FileSystem::{
            ReadFile, FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED,
        };
        use windows_sys::Win32::System::IO::{
            CreateIoCompletionPort, GetQueuedCompletionStatus, OVERLAPPED,
        };

        let mut opened: Vec<Vec<File>> = Vec::new();
        let port =
            unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 0) };
        assert!(!port.is_null(), "completion port");
        for (path, _, _) in files {
            let hs: Vec<File> = (0..handles)
                .map(|_| {
                    std::fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
                        .open(path)
                        .unwrap()
                })
                .collect();
            for h in &hs {
                let p = unsafe { CreateIoCompletionPort(h.as_raw_handle() as _, port, 0, 0) };
                assert_eq!(p, port, "associate a handle with the port");
            }
            opened.push(hs);
        }
        // One request slot per outstanding read: its OVERLAPPED (boxed, so its
        // address is stable for the kernel), its buffer, and which file it reads.
        struct Slot {
            ov: Box<OVERLAPPED>,
            buf: AlignedScratch,
            file: usize,
        }
        let mut slots: Vec<Slot> = (0..files.len() * depth)
            .map(|i| {
                let mut buf = AlignedScratch::new();
                buf.ensure(size).unwrap();
                Slot {
                    ov: Box::new(unsafe { std::mem::zeroed() }),
                    buf,
                    file: i % files.len(),
                }
            })
            .collect();
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut submit = |s: &mut Slot, n: usize| {
            let (_, base, len) = &files[s.file];
            let places = (len - size as u64) / size as u64;
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let at = base + (x >> 16) % places * size as u64;
            *s.ov = unsafe { std::mem::zeroed() };
            s.ov.Anonymous.Anonymous.Offset = at as u32;
            s.ov.Anonymous.Anonymous.OffsetHigh = (at >> 32) as u32;
            let h = opened[s.file][n % handles].as_raw_handle();
            let ok = unsafe {
                ReadFile(
                    h as _,
                    s.buf.as_mut_slice(size).as_mut_ptr(),
                    size as u32,
                    std::ptr::null_mut(),
                    &mut *s.ov,
                )
            };
            if ok == 0 {
                let e = unsafe { GetLastError() };
                assert_eq!(e, ERROR_IO_PENDING, "overlapped ReadFile failed: {e}");
            }
        };
        let t0 = Instant::now();
        for (n, s) in slots.iter_mut().enumerate() {
            submit(s, n);
        }
        let mut outstanding = slots.len();
        let mut bytes = 0u64;
        let mut issued = slots.len();
        while outstanding > 0 {
            let mut got = 0u32;
            let mut key = 0usize;
            let mut done: *mut OVERLAPPED = std::ptr::null_mut();
            let ok =
                unsafe { GetQueuedCompletionStatus(port, &mut got, &mut key, &mut done, u32::MAX) };
            assert!(ok != 0, "a read failed: {}", unsafe { GetLastError() });
            bytes += got as u64;
            outstanding -= 1;
            if t0.elapsed() < window {
                let i = slots
                    .iter()
                    .position(|s| std::ptr::eq(&*s.ov, done))
                    .expect("a completion names one of our requests");
                submit(&mut slots[i], issued);
                issued += 1;
                outstanding += 1;
            }
        }
        let rate = bytes as f64 / t0.elapsed().as_secs_f64() / 1e9;
        unsafe { CloseHandle(port) };
        rate
    }

    /// **How we read, not what we read.** Random reads of the Q3 pack's record
    /// size, and smaller pieces, through overlapped I/O on a completion port —
    /// the mechanism fio and winsat drive NVMe with on Windows — against the
    /// reader threads' blocking reads ([`queue_depth_sweep`]): one handle or
    /// several, one file or two at once. Says whether requests completing one at
    /// a time is the drive or the blocking-read path.
    ///
    /// `cargo test --release -p candle-core --lib direct_io::tests::read_method_sweep -- --ignored --nocapture`
    #[cfg(windows)]
    #[test]
    #[ignore = "reads a multi-GB expert pack off the real drive for ~60 s"]
    fn read_method_sweep() {
        use std::time::Duration;
        let packs = expert_packs();
        let Some(first) = packs.first().cloned() else {
            eprintln!("no model pack with an expert section under ~/.cache/zend/models");
            return;
        };
        let window = Duration::from_secs(3);
        const KIB: usize = 1024;
        let record = 1952 * KIB;
        eprintln!("{}", first.0.display());
        for &(size, depth, handles) in &[
            (record, 1, 1),
            (record, 8, 1),
            (record, 8, 8),
            (record, 32, 1),
            (record, 32, 8),
            (256 * KIB, 64, 1),
            (512 * KIB, 32, 1),
            (1024 * KIB, 32, 1),
            (4096 * KIB, 16, 1),
        ] {
            let rate = overlapped_rate(std::slice::from_ref(&first), size, depth, handles, window);
            eprintln!(
                "  overlapped {:>5} KiB, {depth:>2} outstanding over {handles} handle(s): {rate:>6.2} GB/s",
                size / KIB
            );
        }
        // The reader threads' shape on the overlapped mechanism: N threads,
        // each its own handle and port with one request outstanding — against
        // one thread holding N.
        for threads in [8usize, 16] {
            let rates: Vec<f64> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..threads)
                    .map(|_| {
                        let f = first.clone();
                        s.spawn(move || overlapped_rate(&[f], record, 1, 1, window))
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            eprintln!(
                "  overlapped {:>5} KiB, {threads} threads × 1 outstanding: {:>6.2} GB/s",
                record / KIB,
                rates.iter().sum::<f64>()
            );
        }
        // And the shipped path itself: DirectFile's blocking read on its own
        // handle per thread — alone, beside a buffered handle that has read
        // the header, and beside a memory map of the file, as the model's
        // loader holds one over the same pack for its dense weights.
        let df = DirectFile::open(&first.0).unwrap();
        let buffered = || {
            let mut f = File::open(&first.0).unwrap();
            let mut head = [0u8; 4096];
            std::io::Read::read_exact(&mut f, &mut head).unwrap();
            f
        };
        // SAFETY: the map is only held, never read, and the pack is not
        // written while the test runs.
        let mapped = || unsafe { memmap2::Mmap::map(&File::open(&first.0).unwrap()).unwrap() };
        for (beside, threads) in [
            ("alone", 8usize),
            ("alone", 16),
            ("buffered", 8),
            ("mapped", 8),
            ("mapped, then dropped", 8),
            ("alone", 8),
        ] {
            let _held_file = (beside == "buffered").then(buffered);
            let _held_map = (beside == "mapped").then(mapped);
            if beside == "mapped, then dropped" {
                let m = mapped();
                // Touch a page, as a loader reading a tensor would.
                std::hint::black_box(m[0]);
                drop(m);
            }
            let bytes = std::sync::atomic::AtomicU64::new(0);
            let t0 = std::time::Instant::now();
            let places = (first.2 - record as u64) / record as u64;
            std::thread::scope(|s| {
                for w in 0..threads {
                    let (df, bytes, base) = (&df, &bytes, first.1);
                    s.spawn(move || {
                        let mut buf = AlignedScratch::new();
                        buf.ensure(record).unwrap();
                        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (w as u64 + 1);
                        while t0.elapsed() < window {
                            x = x
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            let at = base + (x >> 16) % places * record as u64;
                            df.read_at_with_handle(w, at, buf.as_mut_slice(record))
                                .unwrap();
                            bytes.fetch_add(record as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                    });
                }
            });
            eprintln!(
                "  DirectFile {:>5} KiB, {threads} threads, {beside}: {:>6.2} GB/s",
                record / KIB,
                bytes.load(std::sync::atomic::Ordering::Relaxed) as f64
                    / t0.elapsed().as_secs_f64()
                    / 1e9
            );
        }
        if let Some(second) = packs.get(1).cloned() {
            eprintln!("  + {}", second.0.display());
            for depth in [8, 16] {
                let rate =
                    overlapped_rate(&[first.clone(), second.clone()], record, depth, 1, window);
                eprintln!(
                    "  overlapped two files at once, {depth} outstanding each: {rate:>6.2} GB/s"
                );
            }
        }
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
