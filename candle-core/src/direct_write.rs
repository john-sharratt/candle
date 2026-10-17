//! A read-write file handle that bypasses the OS page cache.
//!
//! For a file written once and read back cold, if at all — the conversation
//! substrate's redo log, which can grow by gigabytes per conversation. Written
//! through the page cache, every byte of it lands in RAM the OS then has to
//! reclaim, crowding out the expert tiers that host RAM is sized for; and a
//! cached handle open on a file serialises the unbuffered reads made of it
//! through [`DirectFile`](crate::direct_io::DirectFile) (measured on Windows:
//! 6.5 → 4.1 GB/s, `direct_io::tests::read_method_sweep`).
//!
//! The same alignment rules as `direct_io` hold for [`UnbufferedFile::write_at`]
//! and [`UnbufferedFile::read_at`]: the offset, the length and the buffer's
//! address are all multiples of [`DIRECT_IO_SECTOR`]. [`UnbufferedFile::read_span`]
//! reads any range by rounding it out to whole sectors through a scratch.
//!
//! - **Linux**: `O_DIRECT`, positioned `pread` / `pwrite`.
//! - **Windows**: `FILE_FLAG_NO_BUFFERING` on a synchronous handle, positioned
//!   `ReadFile` / `WriteFile`. Appends are sequential, so the overlapped
//!   handles the reader uses to queue many requests at once buy nothing here.
//! - **Other Unix**: buffered, with the same alignment rules, as
//!   `direct_io::open_direct` explains.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use crate::direct_io::{AlignedScratch, DIRECT_IO_SECTOR};

/// How a file is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Create it, failing if it exists; read and write.
    CreateNew,
    /// An existing file, read and write.
    ReadWrite,
    /// An existing file, read only.
    ReadOnly,
}

/// A file opened with the platform's cache-bypassing flag.
pub struct UnbufferedFile {
    file: File,
}

impl UnbufferedFile {
    pub fn open(path: &Path, access: Access) -> io::Result<Self> {
        let mut o = OpenOptions::new();
        o.read(true);
        match access {
            Access::CreateNew => {
                o.write(true).create_new(true);
            }
            Access::ReadWrite => {
                o.write(true);
            }
            Access::ReadOnly => {}
        }
        unbuffered(&mut o);
        Ok(Self {
            file: o.open(path)?,
        })
    }

    /// Write all of `src` at `offset` — both, and `src`'s address, whole
    /// sectors.
    pub fn write_at(&self, offset: u64, src: &[u8]) -> io::Result<()> {
        check_alignment(offset, src.len(), src.as_ptr())?;
        write_all_at(&self.file, offset, src)
    }

    /// Fill `dest` from `offset` — both, and `dest`'s address, whole sectors.
    /// Reading past the end of the file is `UnexpectedEof`.
    pub fn read_at(&self, offset: u64, dest: &mut [u8]) -> io::Result<()> {
        check_alignment(offset, dest.len(), dest.as_ptr())?;
        let got = read_up_to(&self.file, offset, dest)?;
        if got < dest.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "unbuffered read of {} B at {offset} found {got}",
                    dest.len()
                ),
            ));
        }
        Ok(())
    }

    /// Fill `dest` from any `offset`: the whole sectors around it are read
    /// into `scratch` and the range copied out. The file may end inside the
    /// last of them, but not before the range does.
    pub fn read_span(
        &self,
        offset: u64,
        dest: &mut [u8],
        scratch: &mut AlignedScratch,
    ) -> io::Result<()> {
        if dest.is_empty() {
            return Ok(());
        }
        let sector = DIRECT_IO_SECTOR as u64;
        let start = offset / sector * sector;
        let lead = (offset - start) as usize;
        let span = (lead + dest.len()).div_ceil(DIRECT_IO_SECTOR) * DIRECT_IO_SECTOR;
        scratch.ensure(span)?;
        let buf = scratch.as_mut_slice(span);
        let got = read_up_to(&self.file, start, buf)?;
        if got < lead + dest.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "unbuffered read of {} B at {offset}: the file ends {} B in",
                    dest.len(),
                    got.saturating_sub(lead)
                ),
            ));
        }
        dest.copy_from_slice(&buf[lead..lead + dest.len()]);
        Ok(())
    }

    pub fn set_len(&self, len: u64) -> io::Result<()> {
        self.file.set_len(len)
    }

    /// Make what was written durable, data and the length it grew to.
    pub fn sync_data(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    pub fn sync_all(&self) -> io::Result<()> {
        self.file.sync_all()
    }

    pub fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    pub fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }
}

fn check_alignment(offset: u64, len: usize, ptr: *const u8) -> io::Result<()> {
    let s = DIRECT_IO_SECTOR;
    if !(offset as usize).is_multiple_of(s)
        || !len.is_multiple_of(s)
        || !(ptr as usize).is_multiple_of(s)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unbuffered I/O wants whole {s}-byte sectors: offset {offset}, length {len}, \
                 address {ptr:p}"
            ),
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn unbuffered(o: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    o.custom_flags(libc::O_DIRECT);
}

#[cfg(windows)]
fn unbuffered(o: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_NO_BUFFERING;
    o.custom_flags(FILE_FLAG_NO_BUFFERING);
}

/// Unix without `O_DIRECT`: buffered (see the module doc).
#[cfg(all(unix, not(target_os = "linux")))]
fn unbuffered(_o: &mut OpenOptions) {}

#[cfg(unix)]
fn write_all_at(file: &File, offset: u64, src: &[u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(src, offset)
}

#[cfg(windows)]
fn write_all_at(file: &File, offset: u64, src: &[u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < src.len() {
        let n = file.seek_write(&src[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("unbuffered write at {} wrote nothing", offset + done as u64),
            ));
        }
        done += n;
    }
    Ok(())
}

/// One positioned read call takes at most this: whole sectors, and under the
/// Windows API's 32-bit length.
const READ_CALL_MAX: usize = 1 << 30;

/// Read into `dest` from `offset` until it is full or the file ends; the
/// bytes read. A short read is the file's end: reading on from there would
/// start mid-sector, which an unbuffered handle refuses.
fn read_up_to(file: &File, offset: u64, dest: &mut [u8]) -> io::Result<usize> {
    let mut done = 0usize;
    while done < dest.len() {
        let want = (dest.len() - done).min(READ_CALL_MAX);
        let n = read_once(file, offset + done as u64, &mut dest[done..done + want])?;
        done += n;
        if n < want {
            break;
        }
    }
    Ok(done)
}

#[cfg(unix)]
fn read_once(file: &File, offset: u64, dest: &mut [u8]) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(dest, offset)
}

#[cfg(windows)]
fn read_once(file: &File, offset: u64, dest: &mut [u8]) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    match file.seek_read(dest, offset) {
        // Windows reports a read starting at or past the end as this.
        Err(e) if e.raw_os_error() == Some(38) => Ok(0),
        r => r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use std::fs::{remove_file, write};
    use std::path::PathBuf;

    struct Tmp(PathBuf);

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = remove_file(&self.0);
        }
    }

    fn tmp(tag: &str) -> Tmp {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Tmp(temp_dir().join(format!("candle_unbuffered_{tag}_{nanos}.bin")))
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// Whole sectors written at an offset read back byte for byte, through
    /// `read_at` and through a span at an unaligned offset; a reopened file
    /// keeps them.
    #[test]
    fn sectors_written_read_back_aligned_and_unaligned() {
        let path = tmp("roundtrip");
        let s = DIRECT_IO_SECTOR;
        let data = pattern(3 * s, 7);
        let mut src = AlignedScratch::new();
        src.ensure(data.len()).unwrap();
        src.as_mut_slice(data.len()).copy_from_slice(&data);
        {
            let f = UnbufferedFile::open(&path.0, Access::CreateNew).unwrap();
            f.set_len(8 * s as u64).unwrap();
            f.write_at(2 * s as u64, src.as_slice(data.len())).unwrap();
            f.sync_data().unwrap();
        }
        let f = UnbufferedFile::open(&path.0, Access::ReadOnly).unwrap();
        assert_eq!(f.len().unwrap(), 8 * s as u64);
        let mut back = AlignedScratch::new();
        back.ensure(data.len()).unwrap();
        f.read_at(2 * s as u64, back.as_mut_slice(data.len()))
            .unwrap();
        assert_eq!(back.as_slice(data.len()), data.as_slice());
        let mut scratch = AlignedScratch::new();
        let mut piece = vec![0u8; 5000];
        f.read_span(2 * s as u64 + 100, &mut piece, &mut scratch)
            .unwrap();
        assert_eq!(piece.as_slice(), &data[100..5100]);
    }

    /// A misaligned write or read is refused, not attempted; a read past the
    /// end is `UnexpectedEof`; a span may end inside the file's last sector
    /// but not past the file.
    #[test]
    fn misalignment_and_the_end_are_errors() {
        let path = tmp("errors");
        let s = DIRECT_IO_SECTOR;
        let f = UnbufferedFile::open(&path.0, Access::CreateNew).unwrap();
        f.set_len(2 * s as u64).unwrap();
        let mut buf = AlignedScratch::new();
        buf.ensure(2 * s).unwrap();
        let e = f.write_at(1, buf.as_slice(s)).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        let e = f.write_at(0, &buf.as_slice(s)[..100]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        let e = f.read_at(s as u64, buf.as_mut_slice(2 * s)).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
        let mut scratch = AlignedScratch::new();
        let mut tail = vec![0u8; 10];
        f.read_span(2 * s as u64 - 10, &mut tail, &mut scratch)
            .unwrap();
        let e = f
            .read_span(2 * s as u64 - 5, &mut tail, &mut scratch)
            .unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
    }

    /// A file of a length that is not whole sectors — one written buffered —
    /// still reads to its end through a span.
    #[test]
    fn a_span_reads_to_the_end_of_an_unaligned_file() {
        let path = tmp("unaligned_len");
        let data = pattern(6000, 3);
        write(&path.0, &data).unwrap();
        let f = UnbufferedFile::open(&path.0, Access::ReadOnly).unwrap();
        let mut scratch = AlignedScratch::new();
        let mut back = vec![0u8; 1500];
        f.read_span(4500, &mut back, &mut scratch).unwrap();
        assert_eq!(back.as_slice(), &data[4500..]);
    }
}
