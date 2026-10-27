//! A checkpoint read front to back, unbuffered, one window at a time.
//!
//! # Why not the mapping
//!
//! A mapped file hands its pages to the copy in place, which is free when the
//! page cache already holds them. On a
//! member running the engine it does not: WhiteWorld (2026-10-26) had 3.0 GiB
//! of its 64 free and 2.8 GiB of file cache standing, against 10.7 GiB of image
//! checkpoints, so every draw faulted both files in from NVMe a page cluster at
//! a time — 0.5–0.9 GiB/s, 14–23 s of a draw. A buffered read of the same file
//! in 64 MiB pieces made 1.02 GiB/s; an unbuffered one 2.8 GiB/s, and it leaves
//! the cache the engine's own working set stands in alone.
//!
//! So the bytes come off the disk with [`DirectFile`]: sector-aligned, many
//! reads in flight, straight into a buffer the caller supplies — the guest's
//! pinned staging (`super::staging`), so the copy to the card is one DMA from
//! where the disk put the bytes. The final part sector — which a direct read
//! may not run past — is read buffered.

use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;

use candle::direct_io::{round_up_sector, DirectFile, StripeRead, DIRECT_IO_SECTOR};

/// How much of a window one direct read asks for: 16 of these in flight is the
/// reader's whole queue ([`candle::direct_io::MAX_CONCURRENT_READS`]).
const STRIPE: usize = 4 << 20;

/// A checkpoint open for windowed reading.
pub struct Windowed {
    direct: DirectFile,
    tail: File,
    len: u64,
}

/// The buffer bytes `start..end` of a file need: whole sectors from the sector
/// the range starts in.
pub fn window_bytes(start: u64, end: u64) -> usize {
    let sector = DIRECT_IO_SECTOR as u64;
    round_up_sector((end - start / sector * sector) as usize)
}

impl Windowed {
    pub fn open(path: &Path) -> io::Result<Self> {
        let tail = File::open(path)?;
        let len = tail.metadata()?.len();
        Ok(Self {
            direct: DirectFile::open(path)?,
            tail,
            len,
        })
    }

    /// Read the file's bytes `start..end` into `buf`, which is sector-aligned
    /// and at least [`window_bytes`] long; answers where in `buf` they landed.
    pub fn read_into(&mut self, start: u64, end: u64, buf: &mut [u8]) -> io::Result<Range<usize>> {
        if start > end || end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("bytes {start}..{end} of a {}-byte file", self.len),
            ));
        }
        if !(buf.as_ptr() as usize).is_multiple_of(DIRECT_IO_SECTOR)
            || buf.len() < window_bytes(start, end)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "a window of bytes {start}..{end} needs {} sector-aligned bytes, not {}",
                    window_bytes(start, end),
                    buf.len()
                ),
            ));
        }
        let sector = DIRECT_IO_SECTOR as u64;
        let base = start / sector * sector;
        // Whole sectors only: a direct read past the end of the file fails.
        let whole = (round_up_sector(end as usize) as u64).min(self.len / sector * sector);
        let span = (end - base) as usize;

        if whole > base {
            let direct = (whole - base) as usize;
            let mut stripes: Vec<StripeRead<'_>> = buf[..direct]
                .chunks_mut(STRIPE)
                .enumerate()
                .map(|(i, dest)| StripeRead {
                    file_offset: base + (i * STRIPE) as u64,
                    dest,
                })
                .collect();
            self.direct.read_stripes_concurrent(&mut stripes)?;
        }
        if end > whole.max(base) {
            let from = whole.max(base);
            let at = (from - base) as usize;
            read_exact_at(&self.tail, from, &mut buf[at..span])?;
        }
        Ok((start - base) as usize..span)
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, dest: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dest, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut offset: u64, mut dest: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !dest.is_empty() {
        let n = file.seek_read(dest, offset)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        offset += n as u64;
        dest = &mut dest[n..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::direct_io::AlignedScratch;
    use std::io::Write;

    /// `w`'s bytes `start..end`, through an aligned buffer as a caller reads.
    fn read(w: &mut Windowed, start: u64, end: u64) -> io::Result<Vec<u8>> {
        let mut scratch = AlignedScratch::new();
        let len = window_bytes(start, end.max(start));
        scratch.ensure(len)?;
        let buf = scratch.as_mut_slice(len);
        let at = w.read_into(start, end, buf)?;
        Ok(buf[at].to_vec())
    }

    /// A file whose every byte names its own offset, so a misplaced read shows.
    fn numbered(len: usize) -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt.bin");
        let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        (dir, path, bytes)
    }

    /// Every boundary a tensor can sit on: sector-aligned and not, inside one
    /// sector, across stripes, and running to a file end that is not a whole
    /// sector — the part a direct read may not reach.
    #[test]
    fn a_window_reads_exactly_the_bytes_asked_for() {
        let len = 2 * STRIPE + 3 * DIRECT_IO_SECTOR + 123;
        let (_dir, path, bytes) = numbered(len);
        let mut w = Windowed::open(&path).unwrap();
        let s = DIRECT_IO_SECTOR as u64;
        for (start, end) in [
            (0, 1),
            (0, s),
            (5, s - 3),
            (s - 1, s + 1),
            (100, STRIPE as u64 + 77),
            (STRIPE as u64 - 9, 2 * STRIPE as u64 + 9),
            (len as u64 - 123, len as u64),
            (len as u64 - 5000, len as u64),
            (7, len as u64),
            (len as u64, len as u64),
        ] {
            let got = read(&mut w, start, end).unwrap();
            assert_eq!(got, &bytes[start as usize..end as usize], "{start}..{end}");
        }
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        let (_dir, path, _) = numbered(10_000);
        let mut w = Windowed::open(&path).unwrap();
        assert!(read(&mut w, 0, 10_001).is_err());
        assert!(read(&mut w, 9, 8).is_err());
    }

    /// A window's buffer is whole sectors from the sector its range starts in:
    /// one byte past a sector edge takes the next sector too.
    #[test]
    fn a_window_takes_whole_sectors_from_its_first() {
        let s = DIRECT_IO_SECTOR as u64;
        assert_eq!(window_bytes(0, 1), 4096);
        assert_eq!(window_bytes(0, s), 4096);
        assert_eq!(window_bytes(s - 1, s + 1), 8192);
        assert_eq!(window_bytes(s + 5, 3 * s), 8192);
    }

    /// A buffer short of the window, or off a sector boundary, is refused
    /// rather than read past.
    #[test]
    fn a_short_or_misaligned_buffer_is_refused() {
        let (_dir, path, _) = numbered(3 * DIRECT_IO_SECTOR);
        let mut w = Windowed::open(&path).unwrap();
        let mut scratch = AlignedScratch::new();
        scratch.ensure(3 * DIRECT_IO_SECTOR).unwrap();
        let buf = scratch.as_mut_slice(3 * DIRECT_IO_SECTOR);
        assert!(w.read_into(0, 8193, &mut buf[..8192]).is_err());
        assert!(w.read_into(0, 100, &mut buf[1..4097]).is_err());
    }
}
