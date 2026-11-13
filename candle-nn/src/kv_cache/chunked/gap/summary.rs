//! What one gap's kernels tell the host, through mapped memory
//! (`docs/kv_quantize_in_the_gap.md` §3.4).
//!
//! One buffer, because at most one gap is outstanding (§3, §3.8 rule 5): the
//! record fill writes each job's deferred flag and each head's destination
//! record with its eight bands' lease indices; the claim writes the error word
//! if it met a code its table did not cover; and the gap's last launch writes
//! the completion word, behind a system fence, after the convert. The host
//! reads nothing until the completion word holds the gap's sequence number,
//! read as a load of one mapped word at the head of the next gap. No host CUDA
//! call reads or waits on the buffer.

use candle::Result;

#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::ptr::{null_mut, read_volatile, write_bytes, write_volatile};
#[cfg(feature = "cuda")]
use std::sync::atomic::{fence, Ordering};

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::sys;

/// Bands per head: four palettes, K and V.
pub(crate) const BANDS_PER_HEAD: usize = 8;

/// The palettes a head's record names.
const N_PAL: usize = 4;

/// Where everything sits in the summary buffer, for a table of at most
/// `max_jobs` jobs of `n_kv_head` heads at `head_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SummaryLayout {
    pub max_jobs: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

impl SummaryLayout {
    /// Bytes of a head's destination record: `head_dim / 2 + 104`.
    pub(crate) fn record_bytes(&self) -> usize {
        self.head_dim / 2 + 104
    }

    /// Bytes of a head's row: its record, then its bands' lease indices.
    pub(crate) fn head_stride(&self) -> usize {
        self.record_bytes() + BANDS_PER_HEAD * 4
    }

    /// The per-job deferred flags, one `u32` each, from the buffer's start.
    pub(crate) fn flags_off(&self) -> usize {
        0
    }

    /// The first head row, eight-aligned for the record's pointers.
    pub(crate) fn heads_off(&self) -> usize {
        (self.max_jobs * 4).next_multiple_of(8)
    }

    /// The error word the claim sets on a code its table does not cover.
    pub(crate) fn error_off(&self) -> usize {
        self.heads_off() + self.max_jobs * self.n_kv_head * self.head_stride()
    }

    /// The completion word, stored last.
    pub(crate) fn complete_off(&self) -> usize {
        self.error_off() + 4
    }

    pub(crate) fn bytes(&self) -> usize {
        self.complete_off() + 4
    }

    /// Byte offset of `head`'s row (`head` counted across jobs, job-major).
    pub(crate) fn head_off(&self, head: usize) -> usize {
        self.heads_off() + head * self.head_stride()
    }
}

/// One head's row, decoded: the destination record the convert wrote by and
/// each band's lease index.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HeadRow {
    /// Packed palette maps, `head_dim / 4` bytes each.
    pub k_pal: Vec<u8>,
    pub v_pal: Vec<u8>,
    /// Record format codes per palette.
    pub k_fmt: [u8; N_PAL],
    pub v_fmt: [u8; N_PAL],
    pub k_scale: [f32; N_PAL],
    pub v_scale: [f32; N_PAL],
    /// The slot each band was converted into: a leased slot, or the sink for a
    /// deferred job.
    pub k_ptr: [u64; N_PAL],
    pub v_ptr: [u64; N_PAL],
    /// Each band's index in its class's run of the lease, palette-major, K
    /// before V.
    pub lease_idx: [u32; BANDS_PER_HEAD],
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("four bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("eight bytes"))
}

impl HeadRow {
    /// Decode a row from its bytes, laid out as the record fill writes them.
    pub(crate) fn decode(row: &[u8], head_dim: usize) -> Result<Self> {
        let layout = SummaryLayout {
            max_jobs: 0,
            n_kv_head: 0,
            head_dim,
        };
        if row.len() < layout.head_stride() {
            candle::bail!(
                "gap summary: a head row is {} bytes, not {}",
                row.len(),
                layout.head_stride()
            );
        }
        let pal = head_dim / 4;
        let base = head_dim / 2;
        let rec = layout.record_bytes();
        Ok(Self {
            k_pal: row[..pal].to_vec(),
            v_pal: row[pal..2 * pal].to_vec(),
            k_ptr: std::array::from_fn(|p| u64_at(row, base + p * 8)),
            v_ptr: std::array::from_fn(|p| u64_at(row, base + 32 + p * 8)),
            k_fmt: std::array::from_fn(|p| row[base + 64 + p]),
            v_fmt: std::array::from_fn(|p| row[base + 68 + p]),
            k_scale: std::array::from_fn(|p| f32::from_bits(u32_at(row, base + 72 + p * 4))),
            v_scale: std::array::from_fn(|p| f32::from_bits(u32_at(row, base + 88 + p * 4))),
            lease_idx: std::array::from_fn(|b| u32_at(row, rec + b * 4)),
        })
    }
}

/// The completion word for gap `seq`: never zero, which is what a fresh buffer
/// holds.
pub(crate) fn completion_word(seq: u64) -> u32 {
    (seq as u32).wrapping_add(1).max(1)
}

/// The summary buffer, in mapped pinned memory: the kernels write it through
/// its device address and the scheduler thread reads it through its host
/// address once the completion word is in.
#[cfg(feature = "cuda")]
pub(crate) struct MappedSummary {
    layout: SummaryLayout,
    host: *mut u8,
    dev: u64,
}

// SAFETY: written by the gap's kernels and read by the one thread that launched
// them, only after the completion word; at most one gap is outstanding, so the
// buffer is never rewritten while it is read.
#[cfg(feature = "cuda")]
unsafe impl Send for MappedSummary {}

#[cfg(feature = "cuda")]
impl MappedSummary {
    pub(crate) fn new(layout: SummaryLayout) -> Result<Self> {
        let bytes = layout.bytes();
        let mut raw: *mut c_void = null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("gap summary: mapped allocation of {bytes} bytes failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("gap summary: mapped buffer has no device address: {r:?}");
        }
        // SAFETY: `bytes` just allocated. Zero is never a completion word, so
        // the buffer does not read as finished before its first gap.
        unsafe { write_bytes(raw as *mut u8, 0, bytes) };
        Ok(Self {
            layout,
            host: raw as *mut u8,
            dev,
        })
    }

    pub(crate) fn layout(&self) -> SummaryLayout {
        self.layout
    }

    /// The device address of byte `off`.
    pub(crate) fn dev_at(&self, off: usize) -> u64 {
        self.dev + off as u64
    }

    fn word(&self, off: usize) -> *mut u32 {
        assert!(off + 4 <= self.layout.bytes(), "gap summary word {off} out of range");
        // SAFETY: in bounds by the assertion, and four-aligned by the layout.
        unsafe { self.host.add(off) as *mut u32 }
    }

    /// Clear the error word before a gap is launched. Only between gaps: no
    /// kernel of an earlier gap is still running when the host gets here.
    pub(crate) fn arm(&self) {
        // SAFETY: the mapped buffer, live while `self` is.
        unsafe { write_volatile(self.word(self.layout.error_off()), 0) };
    }

    /// Whether the gap that stores `word` has completed. The word is the last
    /// thing the device stores, behind a system fence, so once it reads `word`
    /// every row is visible.
    pub(crate) fn complete(&self, word: u32) -> bool {
        // SAFETY: the mapped buffer, live while `self` is; the device writes it.
        let done = unsafe { read_volatile(self.word(self.layout.complete_off())) } == word;
        if done {
            fence(Ordering::Acquire);
        }
        done
    }

    /// Whether the claim met a code its table did not cover.
    ///
    /// # Safety
    ///
    /// [`Self::complete`] has returned true for the gap being read.
    pub(crate) unsafe fn errored(&self) -> bool {
        read_volatile(self.word(self.layout.error_off())) != 0
    }

    /// Job `job`'s deferred flag.
    ///
    /// # Safety
    ///
    /// As [`Self::errored`], and `job` is inside the gap's table.
    pub(crate) unsafe fn deferred(&self, job: usize) -> bool {
        read_volatile(self.word(self.layout.flags_off() + job * 4)) != 0
    }

    /// Head `head`'s row (`head` counted across jobs, job-major).
    ///
    /// # Safety
    ///
    /// As [`Self::errored`], and `head` is inside the gap's table.
    pub(crate) unsafe fn head_row(&self, head: usize) -> Result<HeadRow> {
        let off = self.layout.head_off(head);
        let len = self.layout.head_stride();
        assert!(off + len <= self.layout.error_off(), "gap summary head {head} out of range");
        HeadRow::decode(
            std::slice::from_raw_parts(self.host.add(off), len),
            self.layout.head_dim,
        )
    }
}

#[cfg(feature = "cuda")]
impl Drop for MappedSummary {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`. The owner drops it only
        // with no gap outstanding, so no kernel still writes it.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_puts_flags_rows_error_and_completion_in_order() {
        let l = SummaryLayout {
            max_jobs: 3,
            n_kv_head: 2,
            head_dim: 128,
        };
        assert_eq!(l.record_bytes(), 168);
        assert_eq!(l.head_stride(), 200);
        assert_eq!(l.flags_off(), 0);
        assert_eq!(l.heads_off(), 16);
        assert_eq!(l.head_off(5), 16 + 5 * 200);
        assert_eq!(l.error_off(), 16 + 6 * 200);
        assert_eq!(l.complete_off(), 16 + 6 * 200 + 4);
        assert_eq!(l.bytes(), 16 + 6 * 200 + 8);
    }

    #[test]
    fn every_row_starts_eight_aligned() {
        for (jobs, hd) in [(1, 128), (7, 128), (5, 256)] {
            let l = SummaryLayout {
                max_jobs: jobs,
                n_kv_head: 3,
                head_dim: hd,
            };
            for h in 0..jobs * 3 {
                assert_eq!(l.head_off(h) % 8, 0, "jobs {jobs} hd {hd} head {h}");
            }
        }
    }

    /// A row as the record fill writes it, byte by byte, decodes field by
    /// field.
    #[test]
    fn a_row_decodes_from_the_bytes_the_fill_writes() {
        let hd = 128;
        let mut row = vec![0u8; hd / 2 + 104 + 32];
        row[0] = 0xE4; // k map: channels 0..4 → palettes 0,1,2,3
        row[hd / 4] = 0x1B; // v map: 3,2,1,0
        let base = hd / 2;
        row[base + 8..base + 16].copy_from_slice(&0x1000_2000u64.to_le_bytes());
        row[base + 32..base + 40].copy_from_slice(&0x3000_4000u64.to_le_bytes());
        row[base + 64 + 2] = 7;
        row[base + 68 + 3] = 15;
        row[base + 72 + 4..base + 72 + 8].copy_from_slice(&0.5f32.to_le_bytes());
        row[base + 88..base + 92].copy_from_slice(&2.0f32.to_le_bytes());
        let rec = hd / 2 + 104;
        row[rec + 4 * 3..rec + 4 * 4].copy_from_slice(&41u32.to_le_bytes());

        let r = HeadRow::decode(&row, hd).unwrap();
        assert_eq!(r.k_pal[0], 0xE4);
        assert_eq!(r.v_pal[0], 0x1B);
        assert_eq!(r.k_pal.len(), 32);
        assert_eq!(r.k_ptr, [0, 0x1000_2000, 0, 0]);
        assert_eq!(r.v_ptr, [0x3000_4000, 0, 0, 0]);
        assert_eq!(r.k_fmt, [0, 0, 7, 0]);
        assert_eq!(r.v_fmt, [0, 0, 0, 15]);
        assert_eq!(r.k_scale, [0.0, 0.5, 0.0, 0.0]);
        assert_eq!(r.v_scale, [2.0, 0.0, 0.0, 0.0]);
        // Band 3 is palette 1's V.
        assert_eq!(r.lease_idx, [0, 0, 0, 41, 0, 0, 0, 0]);
    }

    #[test]
    fn a_short_row_is_refused() {
        assert!(HeadRow::decode(&[0u8; 100], 128).is_err());
    }

    #[test]
    fn the_completion_word_is_never_zero() {
        assert_eq!(completion_word(0), 1);
        assert_eq!(completion_word(u32::MAX as u64), 1);
        assert_eq!(completion_word(41), 42);
    }
}
