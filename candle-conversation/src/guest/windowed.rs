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
//! reads in flight, straight into one reused aligned buffer. The final part
//! sector — which a direct read may not run past — is read buffered.

use std::fs::File;
use std::io;
use std::path::Path;

use candle::direct_io::{
    round_up_sector, AlignedScratch, DirectFile, StripeRead, DIRECT_IO_SECTOR,
};

/// How much of a window one direct read asks for: 16 of these in flight is the
/// reader's whole queue ([`candle::direct_io::MAX_CONCURRENT_READS`]).
const STRIPE: usize = 4 << 20;

/// A checkpoint open for windowed reading.
pub struct Windowed {
    direct: DirectFile,
    tail: File,
    len: u64,
    scratch: AlignedScratch,
}

impl Windowed {
    pub fn open(path: &Path) -> io::Result<Self> {
        let tail = File::open(path)?;
        let len = tail.metadata()?.len();
        Ok(Self {
            direct: DirectFile::open(path)?,
            tail,
            len,
            scratch: AlignedScratch::new(),
        })
    }

    /// The file's bytes `start..end`, read into the window. Valid until the
    /// next call.
    pub fn read(&mut self, start: u64, end: u64) -> io::Result<&[u8]> {
        if start > end || end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("bytes {start}..{end} of a {}-byte file", self.len),
            ));
        }
        let sector = DIRECT_IO_SECTOR as u64;
        let base = start / sector * sector;
        // Whole sectors only: a direct read past the end of the file fails.
        let whole = (round_up_sector(end as usize) as u64).min(self.len / sector * sector);
        let span = (end - base) as usize;
        self.scratch.ensure(span)?;
        let buf = self.scratch.as_mut_slice(round_up_sector(span));

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
        Ok(&self.scratch.as_slice(span)[(start - base) as usize..])
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
    use std::io::Write;

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
            let got = w.read(start, end).unwrap();
            assert_eq!(got, &bytes[start as usize..end as usize], "{start}..{end}");
        }
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        let (_dir, path, _) = numbered(10_000);
        let mut w = Windowed::open(&path).unwrap();
        assert!(w.read(0, 10_001).is_err());
        assert!(w.read(9, 8).is_err());
    }
}
