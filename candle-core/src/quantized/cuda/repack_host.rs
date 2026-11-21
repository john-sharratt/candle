//! Builds the KO twin of a weight from its compact source, and gives the byte
//! sizes the loader plans against.
//!
//! The twin is produced by dequantizing and requantizing the source a band of rows
//! at a time, so the f32 intermediate is bounded by a constant rather than by the
//! tensor, and the load's peak does not grow with the largest weight. A source
//! still on the host is read a band at a time from its mapping. MXFP4 is converted
//! by an exact byte permutation, since no quantize kernel exists for it. The size
//! functions here are the only arithmetic for a twin's bytes, so a caller's carved
//! destination and the writer agree by construction.

use super::*;

/// Whether `dtype` has a GEMX repack kernel, as a property of the dtype alone.
///
/// The storage method above is this function applied to `self.dtype`; the free form exists
/// because the slot planner has to predict `QMatMul::build`'s choice from a GGUF header, with
/// no tensor in hand. A dtype with no repack kernel — a float trunk projection, or MXFP4,
/// neither of which `dtype_to_qtype` names — is dequantized and re-quantized to `Q8_0` before
/// it gets a KO twin, so the twin's width follows `Q8_0`, not the source. A planner that
/// skipped this step would size the slot from a twin that is never built.
pub fn gemx_repacking_supported(dtype: GgmlDType) -> bool {
    if let Ok(qtype) = dtype_to_qtype(dtype) {
        unsafe { is_gemx_supported(qtype as i32) != 0 }
    } else {
        false
    }
}

/// Build a KO twin from a source that is **still on the host**, a band at a time.
///
/// # Why this exists at all
///
/// [`QCudaStorage::repack_ko_into`] bands its f32 intermediate, so the twin costs one band
/// rather than a whole-tensor buffer. But it reads from a source already resident on the
/// device, and getting it there is its own whole-tensor allocation — for a `[248320, 4096]`
/// BF16 `output.weight`, **1,940 MiB** that exists only to be read once, in order, and dropped.
///
/// `dense_span::peak_load_pool_bytes` budgets for exactly that (`largest_source + 2 bands`), so
/// nothing is *wrong* until the largest source is unusually big — and then the margin between
/// the bound and the pool's real behaviour is a few tens of MiB, which allocator granularity
/// and stream-ordered frees from the preceding tensors eat. It presented as
/// `CUDA_ERROR_OUT_OF_MEMORY` on a card with 14 GiB free.
///
/// The source is already in a memory-mapped file. Reading it a band at a time makes the load's
/// device peak `staging band + f32 band + ko band` — about 85 MiB for that head instead of
/// 2,000 — and independent of how large any single tensor is.
///
/// # Contract
///
/// `src` is the tensor's raw bytes as GGUF lays them out: whole blocks, row-major, no padding.
/// `dst`, when given, is at least [`ko_repacked_bytes`] of live device memory that outlives the
/// returned storage — the same contract `repack_ko_into` takes.
pub fn repack_ko_from_host(
    device: &CudaDevice,
    src: &[u8],
    shape: &Shape,
    src_dtype: GgmlDType,
    ko_dtype: GgmlDType,
    dst: Option<(u64, LeaseOrigin)>,
) -> Result<QCudaStorage> {
    let (nrows, ncols) = shape.dims2()?;
    if !crate::quantized::ko_quant::ko_tileable(nrows, ncols) {
        crate::bail!(
            "repack_ko_from_host: shape [{nrows}, {ncols}] must have nrows % 32 == 0 and \
             ncols % 128 == 0"
        );
    }
    // MXFP4 → MXFP4_KO is an exact byte permutation done on the host, not a dequant/requant,
    // so it has no band loop to join. It is also not a shape this path is asked for: the
    // permutation already reads from host bytes and never materialises the source on device.
    if ko_dtype == GgmlDType::MXFP4_KO {
        crate::bail!(
            "repack_ko_from_host: MXFP4_KO is a host-side permutation — use `repack_ko_into`"
        );
    }
    let qtype = dtype_to_qtype(ko_dtype)? as i32;
    let bs = src_dtype.block_size();
    let ts = src_dtype.type_size();
    if !ncols.is_multiple_of(bs) {
        crate::bail!(
            "repack_ko_from_host: {ncols} columns is not a whole number of {src_dtype:?} \
             blocks ({bs})"
        );
    }
    let need = nrows * ncols / bs * ts;
    if src.len() < need {
        crate::bail!(
            "repack_ko_from_host: source holds {} bytes, need {need} for [{nrows}, {ncols}] \
             of {src_dtype:?}",
            src.len()
        );
    }

    let bytes = ko_repacked_bytes(shape, ko_dtype)?;
    let row_groups = nrows / 8;
    let k_blocks = ncols / 128;
    let chunk_bytes = crate::quantized::ko_quant::ko_chunk_bytes(ko_dtype);
    // The same band geometry `repack_ko_into` uses — whole row-groups, sized so the f32
    // intermediate stays inside `REPACK_BAND_BYTES`, and at least one however wide the tensor.
    let band_rows = ((REPACK_BAND_BYTES / (ncols * std::mem::size_of::<f32>())) / 8)
        .max(1)
        .min(row_groups)
        * 8;

    // Three bounded buffers: the staged source bytes, their f32 expansion, and the KO chunks
    // the quantize kernel writes. None scales with the tensor, and all three die with this
    // call, so a load serves them from its scratch at the span's top (`load_scratch`).
    let (mut stage, mut f32_band, mut ko_band) = load_scratch::transient(|| -> Result<_> {
        Ok(unsafe {
            (
                device.alloc::<u8>(band_rows * ncols / bs * ts)?,
                device.alloc::<f32>(band_rows * ncols)?,
                device.alloc::<u8>(band_rows / 8 * k_blocks * chunk_bytes)?,
            )
        })
    })?;

    // **Wrapped before the first fallible step**, for the reason `repack_ko_into` gives at
    // length: `out` may be a lease into the dense block, and a bare `CudaSlice` dropped on an
    // error path frees memory this code does not own. Every `?` below is inside that window.
    let (out, backing) = match dst {
        // SAFETY: the caller's contract — at least `bytes` of live, un-aliased device memory
        // outliving the storage.
        Some((ptr, origin)) => (
            unsafe {
                device
                    .stream_unleased()
                    .upgrade_device_ptr::<u8>(ptr, bytes)
            },
            Backing::Lease(origin),
        ),
        // SAFETY: filled by the loop below.
        None => (unsafe { device.alloc::<u8>(bytes)? }, Backing::Owned),
    };
    let mut me = QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
            inner: out,
            len: bytes,
        }),
        dtype: ko_dtype,
        device: device.clone(),
        backing,
    };
    let stream = device.cuda_stream();

    let mut r0 = 0usize;
    while r0 < nrows {
        let rows = band_rows.min(nrows - r0);
        let src_off = r0 * ncols / bs * ts;
        let src_len = rows * ncols / bs * ts;
        // Host → device for this band only. Every row is a whole number of blocks, so the
        // band's bytes are contiguous in the mapping and need no gather.
        device.memcpy_htod(
            &src[src_off..src_off + src_len],
            &mut stage.slice_mut(..src_len),
        )?;
        {
            let (sp, _sg) = stage.device_ptr(&stream);
            let (fp, _fg) = f32_band.device_ptr_mut(&stream);
            dequantize_f32_into(sp, src_dtype, rows * ncols, fp, &stream)?;
        }
        {
            let (fp, _fg) = f32_band.device_ptr(&stream);
            let (kp, _kg) = ko_band.device_ptr_mut(&stream);
            unsafe {
                run_quantize_ko(
                    fp as *const f32,
                    kp as *mut std::ffi::c_void,
                    rows as i32,
                    ncols as i32,
                    qtype,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        // Scatter the band's chunks to their global positions: one contiguous run per k-block,
        // because `g` is the inner index of `k_blk · row_groups + g`. Identical to
        // `repack_ko_into`'s scatter, and it has to be — the two produce the same bytes.
        let band_groups = rows / 8;
        let g0 = r0 / 8;
        let run = band_groups * chunk_bytes;
        for k in 0..k_blocks {
            let dst_off = (k * row_groups + g0) * chunk_bytes;
            let src_off = k * band_groups * chunk_bytes;
            debug_assert!(
                dst_off + run <= bytes,
                "band scatter passes the destination"
            );
            let src_view = ko_band.slice(src_off..src_off + run);
            let mut dst_view = me.data.inner.slice_mut(dst_off..dst_off + run);
            stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(crate::Error::wrap)?;
        }
        r0 += rows;
    }
    // The H2D copies are stream-ordered against the kernels that read them, and the band
    // buffers free on the same stream when they drop, so no synchronise is needed here — the
    // same reasoning `repack_ko_into` records for its own loop.
    Ok(me)
}

/// Whether `repack_ko_into` can build this source's KO twin **directly**, without the caller
/// first converting it to something else.
///
/// This is the predicate `QMatMul::build` routes on, and the one the slot planner predicts it
/// with — the two must give the same answer or the planner sizes a slot the loader does not
/// write. [`gemx_repacking_supported`] is a narrower question (does this dtype have a GEMX
/// repack kernel) and stays that way, because the streaming path asks it about kernels rather
/// than about routes.
///
/// A **float** source qualifies even though it has no GEMX kernel: `dequantize_f32_into` widens
/// it a band at a time exactly as it dequantizes a block format, so `repack_ko_into` reads it
/// without help. Before that arm existed the only way to give a float projection a KO twin was
/// to dequantize the whole tensor, copy it again through `force_contiguous`, and quantize it to
/// an intermediate `Q8_0` — four whole-tensor buffers, which on a `[248320, 4096]` F16 head is
/// 10,730 MiB to produce a twin the banded route builds inside 48 MiB. It OOMed a 24 GiB card.
pub fn repackable_to_ko(dtype: GgmlDType) -> bool {
    gemx_repacking_supported(dtype)
        || matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
}

/// The f32 band [`QCudaStorage::repack_ko_into`] holds while it repacks.
///
/// **The repack's device scratch, and it does not scale with the tensor.** The repack is
/// dequantize-then-requantize, and it used to compose the two through a whole-tensor f32 buffer
/// — 4,850 MiB for the 27B's `[248320, 5120]` head. That was not merely large but *permanent*:
/// the span reserves its address range once, at a size reduced by exactly that figure, and
/// `cuMemAddressReserve` cannot be grown, so a load-time buffer cost a third of the card for
/// the life of the process. Running a row band at a time makes the intermediate this constant
/// instead, whatever the tensor.
///
/// Public because the load's pool budget has to include it —
/// `dense_span::peak_load_pool_bytes` adds it to the largest source tensor, and a budget that
/// forgets the band OOMs on the first weight rather than at a boundary anyone can read.
pub const REPACK_BAND_BYTES: usize = 48 * 1024 * 1024;

/// Rows in one repack band: whole row-groups, at least one however wide the
/// tensor, never more than the tensor has.
///
/// Extracted from `repack_ko_into` so the bound can be asserted **without a
/// device**. The property that matters — the f32 intermediate is a band and not
/// a whole-tensor buffer — used to be checked by measuring free VRAM across a
/// repack, and `cuMemGetInfo` is device-wide: with `cargo test` running this
/// crate's CUDA tests across threads, a sibling's allocation landed in the delta
/// and was reported as the repack putting its scratch back on the card. It read
/// 4,640 MiB against a 34 MiB twin; the same code alone measures 64 MiB. A
/// number computed here is immune to that, and is the same number the
/// allocation uses.
pub fn repack_band_rows(ncols: usize, row_groups: usize) -> usize {
    ((REPACK_BAND_BYTES / (ncols * std::mem::size_of::<f32>())) / 8)
        .max(1)
        .min(row_groups)
        * 8
}

/// Bytes the f32 intermediate takes for a repack of `nrows × ncols` — the figure
/// [`REPACK_BAND_BYTES`] bounds. See [`repack_band_rows`].
pub fn repack_band_bytes(nrows: usize, ncols: usize) -> usize {
    let row_groups = nrows.div_ceil(8);
    repack_band_rows(ncols, row_groups) * ncols * std::mem::size_of::<f32>()
}

/// Bytes the KO twin of a `shape` weight occupies.
///
/// Knowable before the repack runs, which is what lets a caller carve the
/// destination out of the reservation and hand it to
/// [`QCudaStorage::repack_ko_into`]. A whole number of 8×128 chunks with no
/// `MATRIX_ROW_PADDING` tail — the KO matmul reads chunks, not padded rows.
pub fn ko_repacked_bytes(shape: &Shape, ko_dtype: GgmlDType) -> Result<usize> {
    let (nrows, ncols) = shape.dims2()?;
    if !ko_tileable(nrows, ncols) {
        crate::bail!(
            "ko_repacked_bytes: shape [{nrows}, {ncols}] has no KO twin — nrows % 32 \
             and ncols % 128 are what the matmul tiles on"
        );
    }
    let chunks = (nrows / 8) * (ncols / 128);
    if ko_dtype == GgmlDType::MXFP4_KO {
        return Ok(chunks * crate::quantized::ko_quant::MXFP4_KO_GPU_CHUNK_BYTES);
    }
    Ok(chunks * crate::quantized::ko_quant::ko_chunk_bytes(ko_dtype))
}

/// Zero the [`MATRIX_ROW_PADDING`] tail past a weight's `len`-byte payload.
///
/// The q-matmul kernels read into that tail by design — MMQ's tile loads run
/// past the last row's payload — and multiply it by the activation's zero
/// padding. Zero times a recycled NaN scale is NaN, so a tail left as the
/// allocator found it poisons every output of the call: measured as a 12-row
/// Q2_K matmul whose whole result was NaN once earlier allocations had left
/// non-finite bytes where the tail landed. The tail is the only part zeroed;
/// the payload is written by the copy beside this.
pub(super) fn zero_padding_tail(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    inner: &mut CudaSlice<u8>,
    len: usize,
) -> Result<()> {
    stream.memset_zeros(&mut inner.slice_mut(len..)).w()
}

/// One-shot repack: CPU (GGML bytes) → GPU → repack → GPU → CPU (K/128 bytes).
///
/// Used during swap file creation.  Not performance-critical (one-time cost).
/// Allocates scratch VRAM internally and frees on return.
///
/// Returns the repacked weight bytes in K/128 GEMX format.
pub fn repack_to_host(
    device: &CudaDevice,
    ggml_bytes: &[u8],
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
    target_dtype: GgmlDType,
) -> Result<Vec<u8>> {
    // KO target: dequantize the compact source (`dtype`) and re-quantize to the KO twin
    // (`target_dtype`) ONCE. The host bytes are cached in the expert pinned pool exactly like the
    // gemx repack, so a miss DMA-reloads them with no per-miss re-quant. The int8 KO matmul reads
    // these; the staging pipeline is format-agnostic (gemx K/128 or KO — both just tensors).
    // Already the target format (a prepared / pre-repacked weight, e.g. MXFP4_KO on disk):
    // nothing to do — hand the bytes straight through so the staging path stays uniform and
    // pays no reorder. This is what lets the engine load a pre-KO GGUF with no runtime repack.
    //
    // **This must stay unconditional.** The dtype pair says what the bytes ARE, not whether they
    // still need work: `(X, X)` means "the source is already in the target format". Narrowing
    // this to `is_ko()` targets made the loader repack ALREADY-REPACKED weights a second time —
    // per expert, that is an `alloc` + `alloc_zeros` + H2D + a kernel with an internal
    // `cudaDeviceSynchronize` + D2H into a fresh host `Vec`, ~11,008 times. Measured: 157 GB of
    // private commit on a 189 GB box, the GPU at 6%, and the model gate never finishing.
    //
    // A caller holding RAW GGML that wants the K/128 layout must say so by calling
    // [`repack_gemx_to_host`] directly — the source's provenance (raw vs already-prepared) is
    // the caller's knowledge and cannot be recovered from `(dtype, target_dtype)`.
    if dtype == target_dtype {
        return Ok(ggml_bytes.to_vec());
    }
    // MXFP4_KO: the native MXFP4 experts repack by an EXACT byte-reorder (no dequant/requant),
    // keeping the weights 4-bit. Done on the host straight from the GGUF bytes — the per-sub
    // fold lives in the int8 kernel, so this only permutes nibbles + E8M0 and bakes the per-row
    // e_max scale. (Bypasses the affine `repack_ko` F32-requant route.)
    if target_dtype == GgmlDType::MXFP4_KO {
        if dtype != GgmlDType::MXFP4 {
            crate::bail!("repack_to_host(MXFP4_KO): source must be MXFP4, got {dtype:?}");
        }
        return Ok(crate::quantized::ko_quant::mxfp4_native_to_ko_gpu_chunk(
            ggml_bytes, nrows, ncols,
        ));
    }
    if target_dtype.is_ko() {
        let src = load_repacked(device, ggml_bytes, dtype)?;
        let src_cuda = match &src {
            QStorage::Cuda(s) => s,
            _ => crate::bail!("repack_to_host(KO): expected CUDA storage"),
        };
        let shape: Shape = vec![nrows, ncols].into();
        let ko = src_cuda.repack_ko(&shape, target_dtype)?;
        let host = device
            .memcpy_dtov(&ko.data.inner.slice(..ko.data.len))
            .map_err(crate::Error::wrap)?;
        return Ok(host);
    }

    repack_gemx_to_host(device, ggml_bytes, nrows, ncols, dtype)
}

/// Repack **raw GGML bytes** into the K/128 GEMX layout, keeping the dtype.
///
/// Split out of [`repack_to_host`] because the two cannot be distinguished by their arguments:
/// a GEMX repack keeps the dtype and changes only the byte layout, so it and "the source is
/// already in the target format" both present as `(X, X)`. Which one is meant depends on the
/// **source's provenance** — raw out of a GGUF, or already prepared on disk — which only the
/// caller knows. [`repack_to_host`] therefore treats `(X, X)` as already-prepared (the loader's
/// case, where repacking twice is both wrong and ruinously expensive), and a caller holding raw
/// bytes calls this instead.
///
/// Not performance-critical: one-time, one weight per call, and the underlying kernel carries an
/// internal device sync. Do not use it to convert a whole expert set at load time.
pub fn repack_gemx_to_host(
    device: &CudaDevice,
    ggml_bytes: &[u8],
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
) -> Result<Vec<u8>> {
    let qtype = dtype_to_qtype(dtype)? as i32;

    let supported = unsafe { is_gemx_supported(qtype) };
    if supported == 0 {
        crate::bail!("GEMX repacking not supported for dtype {:?}", dtype);
    }

    let repacked_size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
    if repacked_size < 0 {
        crate::bail!(
            "Failed to get repacked size for {:?} ({nrows}×{ncols})",
            dtype
        );
    }
    let repacked_size = repacked_size as usize;

    // Allocate scratch buffers on the default stream.
    let mut src_buf = unsafe { device.alloc::<u8>(ggml_bytes.len())? };
    let dst_buf = device.alloc_zeros::<u8>(repacked_size)?;

    // H2D: GGML bytes → src VRAM.
    device.memcpy_htod(ggml_bytes, &mut src_buf.slice_mut(..ggml_bytes.len()))?;

    // GPU repack kernel, ordered on the stream the readback below waits for.
    {
        let stream = device.cuda_stream();
        let (src_ptr, _sg) = src_buf.device_ptr(&stream);
        let (dst_ptr, _dg) = dst_buf.device_ptr(&stream);
        let result = unsafe {
            run_repack_gemx(
                src_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                nrows as i32,
                ncols as i32,
                qtype,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if result < 0 {
            crate::bail!("repack_gemx failed for {:?} ({nrows}×{ncols})", dtype);
        }
    }

    // D2H: repacked VRAM → host Vec.
    let repacked = device
        .memcpy_dtov(&dst_buf.slice(..repacked_size))
        .map_err(crate::Error::wrap)?;

    Ok(repacked)
}

/// Get the repacked byte size for a weight matrix of the given dimensions
/// and quantization type, without actually repacking.
pub fn repacked_size_bytes(nrows: usize, ncols: usize, dtype: GgmlDType) -> Result<usize> {
    // KO target: per-128 chunk bytes (the format the expert pinned pool caches for int8 experts).
    if dtype.is_ko() {
        if !nrows.is_multiple_of(8) || !ncols.is_multiple_of(128) {
            crate::bail!("repacked_size_bytes(KO): nrows={nrows} %8, ncols={ncols} %128 required");
        }
        return Ok((nrows / 8) * (ncols / 128) * crate::quantized::ko_quant::ko_chunk_bytes(dtype));
    }
    let qtype = dtype_to_qtype(dtype)? as i32;
    let size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
    if size < 0 {
        crate::bail!("GEMX repacking not supported for dtype {:?}", dtype);
    }
    Ok(size as usize)
}
