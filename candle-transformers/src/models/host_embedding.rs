//! Token embeddings served from pinned host RAM instead of VRAM.
//!
//! A token-embedding table is large and barely touched: on Qwen3-30B-A3B it is
//! 151936 x 2048, costing 594 MiB of VRAM once dequantized to f16, while a
//! forward reads exactly one row per token. That is the worst VRAM-per-access
//! ratio in the model — 30% of the dense weights, for a lookup.
//!
//! So the table stays quantized in host memory, and each forward gathers only
//! the rows it needs:
//!
//! 1. a kernel gathers the selected rows into a contiguous staging buffer,
//!    reading host memory over PCIe (the table lives in a
//!    `cuMemHostAlloc(DEVICEMAP)` allocation, which a kernel can dereference
//!    directly — no VRAM copy);
//! 2. the existing per-format dequantize turns that buffer into the f16 rows the
//!    residual stream expects.
//!
//! The table is *copied* out of the GGUF mmap into that allocation at load, not
//! read from the mmap in place. Pointing the kernel at the mmap is the obvious
//! design and it does not work — see [`HostEmbedding::new`].
//!
//! Step 2 is why the gather is byte-oriented and format-blind: every GGML type
//! stores a row as a whole number of blocks, so a row is `row_bytes` of opaque
//! bytes whatever the quantization, and dequantization is left to
//! `QTensor::dequantize_f16`, which already dispatches ~29 formats with numerics
//! this path must not diverge from. A new quantization type needs no change here.
//!
//! # Why the gather runs on the GPU
//!
//! The token ids are already a device tensor when the embedding is needed. A CPU
//! gather would have to read them back first, and a device-to-host read is a
//! *synchronisation*, not merely a copy: it drains the wave's pipeline at the
//! point the embedding is required, which is the start of the forward.
//!
//! # Cost
//!
//! For a 2048-token prefill of a Q4_K table a row is 1152 B, so a wave moves
//! 2.25 MiB — well under a millisecond of PCIe against a forward measured at
//! 1000-1500 ms on this class of card, because the forward is bound by MoE
//! expert streaming rather than by anything here. Decode gathers one row per
//! sequence, ~11 KiB for a wide batch.

use candle::cuda_backend::Backing;
use candle::quantized::cuda::{alloc_host_mapped, HostMappedAlloc};
use candle::quantized::{cuda::QCudaStorage, GgmlDType, QStorage, QTensor};
use candle::{Device, Result, Tensor};
use candle_kernels::simple::gather_rows::run_gather_rows_bytes;
use cudarc::driver::DevicePtr;

/// Fraction of the card, in percent, above which an embedding table is served
/// from host memory rather than kept resident in VRAM.
///
/// The trade is one-sided in bytes and near-free in time, but not free in
/// complexity, so it applies only where the table is worth reclaiming. Two
/// percent puts the decision where the hardware puts it:
///
/// | card | Qwen3-30B-A3B embedding (f16) | share | resident |
/// |------|------------------------------|-------|----------|
/// | 16 GiB | 594 MiB | 3.63% | host |
/// | 32 GiB | 594 MiB | 1.81% | VRAM |
/// | 2x32 GiB | 594 MiB | 0.91% | VRAM |
///
/// which is the intent: reclaim it on the card short of VRAM, stay out of the
/// way on the card that is not. Expressed against measured sizes rather than a
/// card check, so a larger vocabulary or a different quantization mix moves the
/// decision on its own.
///
/// The judgement is on VRAM alone because the other side of the trade is small
/// and already accounted: the host copy costs the table's QUANTIZED size in
/// pinned RAM (167 MiB against the 594 MiB reclaimed), and `alloc_host_mapped`
/// reports it to the host-RAM budget, so it is structural bytes the budget can
/// see rather than pinned memory it mistakes for pageable.
pub const HOST_EMBEDDING_CAPACITY_PCT: u64 = 2;

/// Whether an embedding of `resident_bytes` (its size **in VRAM**, i.e. after
/// dequantization) should be served from host memory on a card of `capacity`.
///
/// A zero capacity means the governor has not measured the card yet; that is not
/// evidence the table is small, so the resident path is kept.
pub fn should_serve_from_host(resident_bytes: u64, capacity: u64) -> bool {
    if capacity == 0 {
        return false;
    }
    resident_bytes.saturating_mul(100) > capacity.saturating_mul(HOST_EMBEDDING_CAPACITY_PCT)
}

/// Row geometry of a quantized table, and the validation that geometry must pass
/// before a byte gather over it is meaningful.
///
/// Separated from [`HostEmbedding`] because it is pure arithmetic, while the
/// struct proper also resolves a CUDA device address — which needs a registered
/// mapping and a live context, so these rules could otherwise only be exercised
/// on a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    pub n_rows: usize,
    pub ncols: usize,
    /// `(ncols / block_size) * type_size` — identical for every row, because
    /// every GGML type stores whole blocks.
    pub row_bytes: usize,
}

impl RowLayout {
    /// Validate a table of `n_rows` x `ncols` of `dtype` at `byte_offset` in a
    /// mapping of `map_len` bytes.
    ///
    /// Ragged rows would make the byte gather silently wrong, and a table
    /// running past the mapping would have the kernel read unmapped memory, so
    /// both are rejected here rather than discovered mid-forward.
    pub fn new(
        dtype: GgmlDType,
        n_rows: usize,
        ncols: usize,
        byte_offset: usize,
        map_len: usize,
    ) -> Result<Self> {
        let block = dtype.block_size();
        if block == 0 || ncols % block != 0 {
            candle::bail!(
                "host embedding: {ncols} columns is not a whole number of {dtype:?} blocks ({block})"
            );
        }
        let row_bytes = (ncols / block) * dtype.type_size();
        let end = byte_offset
            .checked_add(n_rows.saturating_mul(row_bytes))
            .ok_or_else(|| candle::Error::Msg("host embedding: size overflow".into()))?;
        if end > map_len {
            candle::bail!("host embedding: table ends at {end} but the mmap is {map_len} bytes");
        }
        Ok(Self {
            n_rows,
            ncols,
            row_bytes,
        })
    }

    /// VRAM this table would occupy if dequantized and made resident — the
    /// quantity [`should_serve_from_host`] judges, and the saving realised by not
    /// doing it.
    pub fn resident_bytes_if_dequantized(&self, dtype: candle::DType) -> u64 {
        (self.n_rows as u64)
            .saturating_mul(self.ncols as u64)
            .saturating_mul(dtype.size_in_bytes() as u64)
    }
}

/// A quantized embedding table left in the GGUF mmap, gathered per forward.
pub struct HostEmbedding {
    /// Keeps the pinned host allocation alive; `device_base` points into it.
    _pinned: HostMappedAlloc,
    dtype: GgmlDType,
    layout: RowLayout,
    /// Device address of the table's first row.
    device_base: u64,
}

impl HostEmbedding {
    /// Bind to a table at `byte_offset` within `mmap`.
    ///
    /// The quantized rows are copied once into a `cuMemHostAlloc(DEVICEMAP)`
    /// buffer, which is the allocation CUDA guarantees a kernel can dereference.
    ///
    /// The obvious alternative — point the kernel straight at the GGUF mmap,
    /// which is already `cuMemHostRegister`ed — does not work on WDDM: the
    /// registration reports success on the 17 GB mapping, but
    /// `cuMemHostGetDevicePointer` then refuses it, and forcing the host address
    /// through anyway faults the first forward with
    /// `CUDA_ERROR_ILLEGAL_ADDRESS`. Registered-and-readable are separate
    /// properties there, so this takes the allocation CUDA is explicit about.
    ///
    /// The copy costs the table's QUANTIZED size in pinned host RAM — 167 MiB
    /// for a Q4_K 151936x2048 — to save its DEQUANTIZED size in VRAM, 594 MiB.
    pub fn new(
        mmap: &memmap2::Mmap,
        byte_offset: usize,
        dtype: GgmlDType,
        n_rows: usize,
        ncols: usize,
    ) -> Result<Self> {
        let layout = RowLayout::new(dtype, n_rows, ncols, byte_offset, mmap.len())?;
        let table_bytes = layout.n_rows * layout.row_bytes;
        let (host_ptr, device_base, guard) = alloc_host_mapped(table_bytes)?;
        // SAFETY: `RowLayout::new` checked the whole table lies within the
        // mapping, and `alloc_host_mapped` returned `table_bytes` of storage.
        unsafe {
            std::ptr::copy_nonoverlapping(mmap.as_ptr().add(byte_offset), host_ptr, table_bytes);
        }
        Ok(Self {
            _pinned: guard,
            dtype,
            layout,
            device_base,
        })
    }

    pub fn layout(&self) -> RowLayout {
        self.layout
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    /// Gather the rows named by `ids` and return them dequantized as
    /// `[n_ids, ncols]`.
    ///
    /// `ids` is the live **device** tensor of token ids, read on the device
    /// deliberately — see the module header.
    /// Gather this forward's rows and dequantize them.
    ///
    /// `staging` is the arena the gathered *quantized* bytes are carved from.
    /// They are written by the gather and read once by the dequantize on the next
    /// line, so they are a transient in the strictest sense — and taking them
    /// from a wave span costs nothing, because that span is reserved whether or
    /// not anything uses it.
    ///
    /// The **result** is deliberately not carved from it: the dequantized rows
    /// become `x`, the residual stream, which outlives this call and may be
    /// persisted and resumed on a later wave.
    pub fn embed(&self, ids: &Tensor, device: &Device, staging: Backing) -> Result<Tensor> {
        let cuda = match device {
            Device::Cuda(d) => d,
            _ => candle::bail!("host embedding requires a CUDA device"),
        };
        if ids.dtype() != candle::DType::U32 {
            candle::bail!("host embedding: ids must be U32, got {:?}", ids.dtype());
        }
        let n_ids = ids.elem_count();
        if n_ids == 0 {
            return Tensor::zeros((0, self.layout.ncols), candle::DType::F16, device);
        }

        let (ids_storage, ids_layout) = ids.storage_and_layout();
        let ids_slice = match &*ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("host embedding: ids must be a cuda tensor"),
        }
        .slice(ids_layout.start_offset()..);

        let elem_count = n_ids * self.layout.ncols;
        // Uninitialised, not zeroed: the gather below writes every byte of the
        // data region — `RowLayout` requires `ncols` to be a whole number of
        // blocks, so `n_ids * row_bytes` is exactly `len` — and the dequantize
        // that consumes it reads exactly `elem_count` elements, never the
        // padding tail. Zeroing first would memset the whole staging buffer
        // only for the gather to overwrite it on the next launch.
        //
        // SAFETY: `run_gather_rows_bytes` fills all `n_ids * row_bytes` bytes
        // before `dequantize_f16` reads them, and this storage never reaches a
        // q-matmul kernel, which is the path that over-reads into the padding.
        let staged = unsafe { QCudaStorage::uninit_from(cuda, elem_count, self.dtype, staging)? };

        let stream = cuda.cuda_stream();
        let (ids_ptr, _ids_guard) = ids_slice.device_ptr(&stream);
        unsafe {
            run_gather_rows_bytes(
                self.device_base as *const std::ffi::c_void,
                ids_ptr as *const u32,
                staged.data_ptr() as *mut std::ffi::c_void,
                self.layout.row_bytes as i64,
                self.layout.n_rows as i64,
                n_ids as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        // Hand the gathered rows to the ordinary quantized path: `dequantize_f16`
        // is the same call the resident embedding would have made at load, so
        // every format it supports is supported here with identical numerics.
        let qt = QTensor::new(QStorage::Cuda(staged), (n_ids, self.layout.ncols))?;
        qt.dequantize_f16(device)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial as gpu_guard;
    use half::f16;

    const BLOCK: usize = 32;

    /// A Q8_0 table with a distinct value in every slot, plus the exact `f16`
    /// each slot must dequantize to.
    ///
    /// Distinctness is the point: a gather that dropped a row, doubled one, or
    /// left a byte unwritten has to change a value that is checked, so no error
    /// can hide behind a neighbour that happened to hold the same number.
    fn synthetic_q8_0(n_rows: usize, ncols: usize) -> (Vec<u8>, Vec<Vec<f16>>) {
        let mut bytes = Vec::with_capacity(n_rows * (ncols / BLOCK) * (2 + BLOCK));
        let mut expected = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let mut row = Vec::with_capacity(ncols);
            for b in 0..ncols / BLOCK {
                let scale = f16::from_f32(0.25 + (r * 3 + b) as f32 * 0.125);
                bytes.extend_from_slice(&scale.to_le_bytes());
                for i in 0..BLOCK {
                    let q = (((r * 7 + b * 13 + i * 3) % 251) as i32 - 125) as i8;
                    bytes.push(q as u8);
                    row.push(f16::from_f32(scale.to_f32() * q as f32));
                }
            }
            expected.push(row);
        }
        (bytes, expected)
    }

    /// The gather must reproduce the named rows exactly — asserted on the `f16`
    /// values themselves rather than a tolerance, since Q8_0 dequantization is
    /// `scale * q` and has one right answer.
    ///
    /// This is also what covers `embed`'s uninitialised staging buffer: the
    /// gather is trusted to write every byte it allocates, and a byte it missed
    /// would surface here as whatever the driver last left there.
    #[test]
    fn embed_gathers_exact_rows_from_the_host_table() -> Result<()> {
        let _gpu = gpu_guard();
        let device = Device::new_cuda(0)?;
        let (n_rows, ncols) = (8usize, 64usize);
        let (bytes, expected) = synthetic_q8_0(n_rows, ncols);

        let path = std::env::temp_dir().join("candle_host_embedding_gather.bin");
        std::fs::write(&path, &bytes).map_err(candle::Error::wrap)?;
        let file = std::fs::File::open(&path).map_err(candle::Error::wrap)?;
        // SAFETY: the file is written above and not modified while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(candle::Error::wrap)? };
        let table = HostEmbedding::new(&mmap, 0, GgmlDType::Q8_0, n_rows, ncols)?;

        // Repeats and a non-monotonic order, so a gather that ignored the ids
        // and copied a contiguous span cannot pass.
        let ids = [3u32, 1, 7, 3, 0];
        // No wave in a unit test, so the staging is an ordinary allocation —
        // which is exactly the path a caller without a generation takes.
        let out = table.embed(
            &Tensor::new(ids.as_slice(), &device)?,
            &device,
            Backing::Owned,
        )?;
        assert_eq!(out.dims(), &[ids.len(), ncols]);
        let got = out.to_vec2::<f16>()?;

        for (k, &id) in ids.iter().enumerate() {
            assert_eq!(
                got[k], expected[id as usize],
                "row {k} should be table row {id}"
            );
        }

        drop(mmap);
        std::fs::remove_file(&path).ok();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The policy must reclaim on the 16 GiB card and leave the 32 GiB card
    /// alone — the point of expressing it as a fraction of capacity rather than
    /// a hardcoded card check.
    #[test]
    fn policy_reclaims_on_a_tight_card_and_not_a_roomy_one() {
        const MIB: u64 = 1 << 20;
        let embedding = 594 * MIB;
        assert!(should_serve_from_host(embedding, 16376 * MIB));
        assert!(!should_serve_from_host(embedding, 32768 * MIB));
        assert!(!should_serve_from_host(embedding, 65536 * MIB));
    }

    /// A bigger vocabulary moves the decision on its own, with nobody re-tuning
    /// a constant per model.
    #[test]
    fn policy_follows_the_table_not_the_model_name() {
        const MIB: u64 = 1 << 20;
        assert!(!should_serve_from_host(594 * MIB, 32768 * MIB));
        assert!(should_serve_from_host(1188 * MIB, 32768 * MIB));
    }

    /// Before the governor has measured the card, an unmeasured capacity is not
    /// evidence the table is small — keep the resident path.
    #[test]
    fn policy_holds_the_resident_path_without_a_measurement() {
        assert!(!should_serve_from_host(u64::MAX, 0));
    }

    /// Exactly at the threshold is not over it.
    #[test]
    fn policy_threshold_is_exclusive() {
        let capacity = 100_000u64;
        let exactly_two_pct = capacity * HOST_EMBEDDING_CAPACITY_PCT / 100;
        assert!(!should_serve_from_host(exactly_two_pct, capacity));
        assert!(should_serve_from_host(exactly_two_pct + 1, capacity));
    }

    /// Q8_0 is 32 elements per 34-byte block, so 64 columns is 2 blocks = 68
    /// bytes per row — deliberately neither a power of two nor 16-byte aligned,
    /// so a stride mistake cannot hide behind a tidy number.
    #[test]
    fn row_bytes_is_whole_blocks() {
        let l = RowLayout::new(GgmlDType::Q8_0, 8, 64, 0, 8 * 68).expect("well formed");
        assert_eq!(l.row_bytes, 68);
        assert_eq!(l.n_rows, 8);
        assert_eq!(l.ncols, 64);
    }

    /// Ragged rows would make the byte gather silently wrong.
    #[test]
    fn a_column_count_that_is_not_whole_blocks_is_rejected() {
        let err = match RowLayout::new(GgmlDType::Q8_0, 2, 63, 0, 1 << 20) {
            Ok(_) => panic!("63 is not a multiple of the 32-element Q8_0 block"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("whole number"), "{err}");
    }

    /// A table running past the mapping would have the kernel read unmapped
    /// memory; reject it at construction instead.
    #[test]
    fn a_table_larger_than_the_mapping_is_rejected() {
        let err = match RowLayout::new(GgmlDType::Q8_0, 1000, 64, 0, 8 * 68) {
            Ok(_) => panic!("1000 rows do not fit"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("mmap is"), "{err}");
    }

    /// The offset counts against the mapping, not just the table size.
    #[test]
    fn the_offset_is_included_in_the_bounds_check() {
        assert!(RowLayout::new(GgmlDType::Q8_0, 8, 64, 13, 13 + 8 * 68).is_ok());
        assert!(RowLayout::new(GgmlDType::Q8_0, 8, 64, 14, 13 + 8 * 68).is_err());
    }

    /// The policy judges the DEQUANTIZED size — what the table would have
    /// occupied in VRAM, not what it occupies in the file.
    #[test]
    fn resident_size_is_the_dequantized_size() {
        let l = RowLayout::new(GgmlDType::Q8_0, 4, 64, 0, 4 * 68).expect("well formed");
        assert_eq!(l.resident_bytes_if_dequantized(candle::DType::F16), 512);
        assert_eq!(l.resident_bytes_if_dequantized(candle::DType::F32), 1024);
    }
}
