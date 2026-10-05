//! The sampling kernel's per-row count tables, resident on the device.
//!
//! The kernel reads a row's frequency/presence counts and its cross-turn counts
//! densely — `counts[row * vocab + token]` — while a turn has said a few hundred
//! distinct tokens out of a vocabulary of a quarter of a million. Rebuilding the
//! dense `[rows, vocab]` image on the host and uploading it on every dispatch
//! moved megabytes across the bus to carry kilobytes of information, and the
//! speculative accept walk dispatches once per block position.
//!
//! The table here stays on the device and is all zero between dispatches. A
//! dispatch uploads its nonzero entries once, offsets then values, and stamps
//! them in ([`DeviceCountTable::stamp`]); the kernel reads the table; and the
//! same offsets are written back to zero ([`DeviceCountTable::clear`]). On CUDA
//! both writes are one launch of `run_stamp_counts` spread over the entries,
//! with nothing allocated for the clear; elsewhere they are `scatter_set`.

use candle::cuda_backend::CudaStorageSlice;
use candle::{DType, Device, Layout, Result, Storage, Tensor};
use candle_kernels::sampling::run_stamp_counts;
use cudarc::driver::{CudaStream, DevicePtr, SyncOnDrop};
use std::ffi::c_void;

/// Consecutive dispatches, each fitting in a quarter of the held table, after
/// which the table is reallocated at the size they need. One wide wave grows
/// the table to its width; a long run of narrow ones gives that ground back.
const SHRINK_AFTER: usize = 1024;

/// One dispatch's nonzero counts as flat table offsets (`row * vocab + token`)
/// and the counts stored there.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct SparseCounts {
    pub offsets: Vec<u32>,
    pub values: Vec<u32>,
}

impl SparseCounts {
    /// Gather each row's nonzero entries, rows in dispatch order.
    ///
    /// A row is `None` when its penalties are off for this dispatch (a row
    /// writing a tool call), which leaves its table row reading zero. Otherwise
    /// it is the tokens that may hold a nonzero count and the dense counts they
    /// index; a listed token whose count is zero contributes nothing.
    pub fn gather<'a>(
        vocab: usize,
        rows: impl IntoIterator<Item = Option<(&'a [u32], &'a [i32])>>,
    ) -> Self {
        let mut out = Self::default();
        for (r, row) in rows.into_iter().enumerate() {
            let Some((tokens, counts)) = row else {
                continue;
            };
            for &t in tokens {
                debug_assert!(
                    (t as usize) < vocab,
                    "token {t} is outside the {vocab}-entry vocabulary"
                );
                let c = counts[t as usize];
                if c > 0 {
                    out.offsets.push((r * vocab) as u32 + t);
                    out.values.push(c as u32);
                }
            }
        }
        out
    }

    /// The offsets followed by the values, as the one buffer a dispatch uploads.
    fn packed(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.offsets.len() * 2);
        words.extend_from_slice(&self.offsets);
        words.extend_from_slice(&self.values);
        words
    }
}

/// A `[rows, vocab]` count table that reads zero everywhere a dispatch did not
/// stamp. Grows to the widest dispatch seen, and shrinks back after
/// [`SHRINK_AFTER`] dispatches that each need under a quarter of it.
pub(crate) struct DeviceCountTable {
    vocab: usize,
    table: Option<Tensor>,
    narrow_run: usize,
    /// Where a dispatch's packed entries are uploaded on CUDA: held and grown
    /// to the widest dispatch seen, so a stamp allocates nothing. The clear
    /// that ends a dispatch reads it on the same stream before the next
    /// stamp's upload overwrites it.
    entries: Option<Tensor>,
}

/// A table with one dispatch's counts stamped in. Hand it back to
/// [`DeviceCountTable::clear`] once the kernel reading it has been enqueued.
pub(crate) struct Stamped {
    table: Tensor,
    /// `[2n]` `U32`: the `n` offsets stamped, then their values. `None` when
    /// the dispatch stamped nothing.
    entries: Option<(Tensor, usize)>,
}

impl Stamped {
    /// The dense table the kernel reads, `U32` over `rows * vocab` entries;
    /// the counts are non-negative, so its bytes are the kernel's `int32_t`.
    pub fn table(&self) -> &Tensor {
        &self.table
    }
}

impl DeviceCountTable {
    pub fn new(vocab: usize) -> Self {
        Self {
            vocab,
            table: None,
            narrow_run: 0,
            entries: None,
        }
    }

    /// The table, at least `rows` rows deep, with `counts` stamped in and zero
    /// everywhere else.
    ///
    /// A failed write leaves no table behind: the next dispatch allocates a
    /// fresh zero one rather than reading this one's partial stamp.
    pub fn stamp(
        &mut self,
        device: &Device,
        rows: usize,
        counts: &SparseCounts,
    ) -> Result<Stamped> {
        let table = self.sized(device, rows)?;
        let entries = if counts.offsets.is_empty() {
            None
        } else {
            let n = counts.offsets.len();
            let entries = match device {
                Device::Cuda(_) => {
                    let held = self.entries_for(device, 2 * n)?;
                    upload_words(&held, &counts.packed())?;
                    held
                }
                _ => Tensor::from_vec(counts.packed(), 2 * n, device)?,
            };
            if let Err(e) = write_entries(&table, &entries, n, true) {
                self.table = None;
                return Err(e);
            }
            Some((entries, n))
        };
        Ok(Stamped { table, entries })
    }

    /// The held entries buffer as a `[words]` view, grown by doubling when a
    /// dispatch outgrows it.
    fn entries_for(&mut self, device: &Device, words: usize) -> Result<Tensor> {
        if !self
            .entries
            .as_ref()
            .is_some_and(|e| e.elem_count() >= words)
        {
            let grown = self
                .entries
                .as_ref()
                .map_or(0, |e| e.elem_count() * 2)
                .max(words);
            // Fully written by the upload before any launch reads it.
            self.entries = Some(Tensor::empty(grown, DType::U32, device)?);
        }
        self.entries
            .as_ref()
            .expect("sized above")
            .narrow(0, 0, words)
    }

    /// Return the stamped entries to zero, so the next dispatch starts from an
    /// all-zero table. A failed clear drops the table for the same reason a
    /// failed stamp does.
    pub fn clear(&mut self, stamped: Stamped) -> Result<()> {
        let Some((entries, n)) = stamped.entries else {
            return Ok(());
        };
        let cleared = write_entries(&stamped.table, &entries, n, false);
        if cleared.is_err() {
            self.table = None;
        }
        cleared
    }

    /// The held table, reallocated when `rows` outgrows it or when a long run
    /// of narrow dispatches has left most of it unused.
    fn sized(&mut self, device: &Device, rows: usize) -> Result<Tensor> {
        let fit = rows.max(1).next_power_of_two() * self.vocab;
        let held = self.table.as_ref().map_or(0, Tensor::elem_count);
        let realloc = if held < rows * self.vocab {
            true
        } else if held >= 4 * fit {
            self.narrow_run += 1;
            self.narrow_run >= SHRINK_AFTER
        } else {
            self.narrow_run = 0;
            false
        };
        if realloc {
            if fit > u32::MAX as usize {
                candle::bail!(
                    "count table of {} rows × {} vocab overflows the u32 offsets that address it",
                    fit / self.vocab,
                    self.vocab
                );
            }
            self.narrow_run = 0;
            // Zeroed, not uninitialised: the zero is what every unstamped entry
            // reads, and the table is kept zero from here on by `clear`.
            self.table = Some(Tensor::zeros(fit, DType::U32, device)?);
        }
        Ok(self.table.clone().expect("table sized above"))
    }
}

/// Write `entries`' `n` offsets into `table`: their values when `values`,
/// otherwise zero.
fn write_entries(table: &Tensor, entries: &Tensor, n: usize, values: bool) -> Result<()> {
    if let Device::Cuda(dev) = table.device() {
        let stream = dev.cuda_stream();
        let (t_storage, t_layout) = table.storage_and_layout();
        let (e_storage, e_layout) = entries.storage_and_layout();
        let (t_ptr, _t_guard) = count_table_ptr(&t_storage, t_layout, &stream)?;
        let (e_ptr, _e_guard) = count_table_ptr(&e_storage, e_layout, &stream)?;
        let values_ptr = if values {
            (e_ptr + n as u64 * 4) as *const u32
        } else {
            std::ptr::null()
        };
        // SAFETY: `entries` holds `2n` u32 words — `n` distinct offsets inside
        // `table`, then their values — and both buffers stay alive, guarded on
        // `stream`, while the launch is queued on it.
        unsafe {
            run_stamp_counts(
                e_ptr as *const u32,
                values_ptr,
                n as u32,
                t_ptr as *mut u32,
                stream.cu_stream() as *mut c_void,
            );
        }
        return Ok(());
    }
    let offsets = entries.narrow(0, 0, n)?;
    let written = if values {
        entries.narrow(0, n, n)?
    } else {
        Tensor::zeros(n, DType::U32, entries.device())?
    };
    table.scatter_set(&offsets, &written, 0)
}

/// Upload `words` into `dst`, a held CUDA `U32` buffer of exactly that many
/// elements, on its stream — the copy lands in place and allocates nothing.
fn upload_words(dst: &Tensor, words: &[u32]) -> Result<()> {
    let Device::Cuda(dev) = dst.device() else {
        candle::bail!("count table entries must be on CUDA");
    };
    if dst.elem_count() != words.len() {
        candle::bail!(
            "count table entries: {} words into a {}-word view",
            words.len(),
            dst.elem_count()
        );
    }
    let stream = dev.cuda_stream();
    let (storage, layout) = dst.storage_and_layout();
    let (ptr, _guard) = count_table_ptr(&storage, layout, &stream)?;
    // SAFETY: `ptr` addresses `words.len()` u32s of `dst`'s own storage (the
    // length is checked above), and the copy is ordered on the stream every
    // reader of `dst` runs on. A pageable source is staged before this returns.
    unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            ptr,
            words.as_ptr() as *const c_void,
            std::mem::size_of_val(words),
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| candle::Error::Msg(format!("count table entries upload: {e}")))?;
    }
    Ok(())
}

/// The device address of a resident `U32` buffer at its view's offset, with
/// the stream guard to hold across the launch that reads it.
pub(crate) fn count_table_ptr<'a>(
    storage: &'a Storage,
    layout: &Layout,
    stream: &'a CudaStream,
) -> Result<(u64, SyncOnDrop<'a>)> {
    match storage {
        Storage::Cuda(cs) => match &cs.slice {
            CudaStorageSlice::U32(s) => {
                let (ptr, guard) = s.device_ptr(stream);
                Ok((ptr + layout.start_offset() as u64 * 4, guard))
            }
            _ => candle::bail!("sampler count table must be U32"),
        },
        _ => candle::bail!("sampler count table must be on CUDA"),
    }
}

/// The two tables a sampling dispatch reads.
pub(crate) struct PenaltyTables {
    pub tokens: DeviceCountTable,
    pub cross: DeviceCountTable,
}

impl PenaltyTables {
    pub fn new(vocab: usize) -> Self {
        Self {
            tokens: DeviceCountTable::new(vocab),
            cross: DeviceCountTable::new(vocab),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_flattens_each_rows_nonzero_entries_and_skips_switched_off_rows() {
        let row0 = [0, 3, 0, 1, 0];
        let row2 = [2, 0, 0, 0, 7];
        let rows: [Option<(&[u32], &[i32])>; 3] =
            [Some((&[3, 1], &row0)), None, Some((&[4, 0, 2], &row2))];
        let got = SparseCounts::gather(5, rows);
        // Row 1 is off; row 2 lists token 2, whose count is zero.
        assert_eq!(
            got,
            SparseCounts {
                offsets: vec![3, 1, 14, 10],
                values: vec![1, 3, 7, 2],
            }
        );
    }

    #[test]
    fn a_dispatch_uploads_its_offsets_then_its_values() {
        let counts = SparseCounts {
            offsets: vec![3, 1, 14],
            values: vec![1, 3, 7],
        };
        assert_eq!(counts.packed(), vec![3, 1, 14, 1, 3, 7]);
    }

    #[test]
    fn a_stamped_table_reads_the_counts_and_clears_back_to_zero() -> Result<()> {
        stamp_and_clear_on(&Device::Cpu)
    }

    /// The same through the CUDA stamp kernel, on the device the sampler reads
    /// it from.
    #[test]
    fn a_stamped_cuda_table_reads_the_counts_and_clears_back_to_zero() -> Result<()> {
        match Device::new_cuda(0) {
            Ok(dev) => stamp_and_clear_on(&dev),
            Err(_) => Ok(()), // No CUDA device on this box.
        }
    }

    fn stamp_and_clear_on(dev: &Device) -> Result<()> {
        let dev = dev.clone();
        let mut table = DeviceCountTable::new(4);
        let counts = SparseCounts {
            offsets: vec![1, 6],
            values: vec![5, 2],
        };
        let stamped = table.stamp(&dev, 2, &counts)?;
        assert_eq!(
            stamped.table().to_vec1::<u32>()?,
            vec![0, 5, 0, 0, 0, 0, 2, 0]
        );
        let held = stamped.table().clone();
        table.clear(stamped)?;
        assert_eq!(held.to_vec1::<u32>()?, vec![0; 8]);
        Ok(())
    }

    /// A dispatch uploads into the held entries buffer: a narrower one after a
    /// wider one reuses it, and the counts it stamps are still exact.
    #[test]
    fn a_cuda_dispatch_reuses_the_held_entries_buffer() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else {
            return Ok(()); // No CUDA device on this box.
        };
        let mut table = DeviceCountTable::new(4);
        let wide = SparseCounts {
            offsets: vec![0, 1, 2, 3],
            values: vec![1, 2, 3, 4],
        };
        let stamped = table.stamp(&dev, 1, &wide)?;
        table.clear(stamped)?;
        let held = table
            .entries
            .clone()
            .expect("a CUDA stamp holds its entries");
        let narrow = SparseCounts {
            offsets: vec![2],
            values: vec![9],
        };
        let stamped = table.stamp(&dev, 1, &narrow)?;
        assert_eq!(stamped.table().to_vec1::<u32>()?, vec![0, 0, 9, 0]);
        assert!(
            table
                .entries
                .as_ref()
                .is_some_and(|e| e.same_storage(&held)),
            "the narrower dispatch must upload into the same buffer"
        );
        table.clear(stamped)?;
        Ok(())
    }

    /// More entries than one 256-thread block, so the stamp spans the grid.
    #[test]
    fn a_cuda_stamp_wider_than_one_block_lands_every_entry() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else {
            return Ok(()); // No CUDA device on this box.
        };
        let mut table = DeviceCountTable::new(1000);
        let offsets: Vec<u32> = (0..600).map(|i| i * 3).collect();
        let values: Vec<u32> = (0..600).map(|i| i + 1).collect();
        let counts = SparseCounts {
            offsets: offsets.clone(),
            values: values.clone(),
        };
        let stamped = table.stamp(&dev, 2, &counts)?;
        let got = stamped.table().to_vec1::<u32>()?;
        let mut want = vec![0u32; 2000];
        for (&o, &v) in offsets.iter().zip(&values) {
            want[o as usize] = v;
        }
        assert_eq!(got, want);
        let held = stamped.table().clone();
        table.clear(stamped)?;
        assert_eq!(held.to_vec1::<u32>()?, vec![0; 2000]);
        Ok(())
    }

    #[test]
    fn the_table_grows_to_a_wider_dispatch_and_starts_it_from_zero() -> Result<()> {
        let dev = Device::Cpu;
        let mut table = DeviceCountTable::new(2);
        let first = table.stamp(
            &dev,
            1,
            &SparseCounts {
                offsets: vec![1],
                values: vec![4],
            },
        )?;
        assert_eq!(first.table().to_vec1::<u32>()?, vec![0, 4]);
        table.clear(first)?;
        // Three rows round up to four.
        let wide = table.stamp(
            &dev,
            3,
            &SparseCounts {
                offsets: vec![5],
                values: vec![1],
            },
        )?;
        assert_eq!(wide.table().to_vec1::<u32>()?, vec![0, 0, 0, 0, 0, 1, 0, 0]);
        table.clear(wide)?;
        Ok(())
    }

    #[test]
    fn a_long_narrow_run_gives_the_wide_tables_ground_back() -> Result<()> {
        let dev = Device::Cpu;
        let mut table = DeviceCountTable::new(2);
        let wide = table.stamp(&dev, 8, &SparseCounts::default())?;
        assert_eq!(wide.table().elem_count(), 16);
        table.clear(wide)?;
        for _ in 1..SHRINK_AFTER {
            let narrow = table.stamp(&dev, 1, &SparseCounts::default())?;
            assert_eq!(narrow.table().elem_count(), 16);
            table.clear(narrow)?;
        }
        let shrunk = table.stamp(
            &dev,
            1,
            &SparseCounts {
                offsets: vec![1],
                values: vec![9],
            },
        )?;
        assert_eq!(shrunk.table().to_vec1::<u32>()?, vec![0, 9]);
        table.clear(shrunk)?;
        Ok(())
    }

    #[test]
    fn a_dispatch_at_half_the_table_keeps_it() -> Result<()> {
        let dev = Device::Cpu;
        let mut table = DeviceCountTable::new(2);
        let wide = table.stamp(&dev, 4, &SparseCounts::default())?;
        table.clear(wide)?;
        for _ in 0..SHRINK_AFTER + 1 {
            let half = table.stamp(&dev, 2, &SparseCounts::default())?;
            assert_eq!(half.table().elem_count(), 8);
            table.clear(half)?;
        }
        Ok(())
    }

    #[test]
    fn an_empty_dispatch_stamps_nothing() -> Result<()> {
        let mut table = DeviceCountTable::new(3);
        let stamped = table.stamp(&Device::Cpu, 1, &SparseCounts::default())?;
        assert_eq!(stamped.table().to_vec1::<u32>()?, vec![0, 0, 0]);
        table.clear(stamped)?;
        Ok(())
    }
}
