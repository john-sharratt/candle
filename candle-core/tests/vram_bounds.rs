//! The repack band bounds — measured against the whole card, so they need it.
//!
//! Each test here brackets an operation with [`candle_core::quantized::get_vram_info`],
//! which reports free memory for the *device*. Any allocation anybody else makes
//! inside that window lands in the delta, so these cannot share a process with
//! the rest of the CUDA suite: in `candle-core --lib` they measured a 1,226 MiB
//! dip against a 258 MiB entitlement while 180 sibling tests allocated
//! concurrently, and 184 of 184 passed under `--test-threads=1`.
//!
//! **Loosening the bound was the wrong repair.** What they pin is that a
//! whole-tensor intermediate has not come back — the regression that reserved
//! 4,850 MiB permanently, and the one that OOMed a 24 GiB card with 14 GiB
//! free. A threshold widened until a contended run passes would stop detecting
//! either.
//!
//! So they live in their own test binary, where the only allocations in the
//! window are their own, and the mutex below keeps the three of them from
//! measuring through each other.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use candle_core::quantized::{cuda, get_vram_info, ko_quant, GgmlDType, QStorage, QTensor};
use candle_core::{CudaDevice, DType, Device, Error, Result, Shape, Tensor};

static VRAM_MEASUREMENT: Mutex<()> = Mutex::new(());

/// Hold the card for the length of a measurement.
///
/// A poisoned lock is taken anyway: a test that already panicked has reported
/// itself, and failing its siblings on top of that only hides which one it was.
fn measuring_vram() -> MutexGuard<'static, ()> {
    VRAM_MEASUREMENT
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// The card, both ways round: `CudaDevice::new` is crate-private, so a test
/// outside the crate reaches it through the public `Device` and takes the inner
/// handle back out for the stream and copy calls.
fn card() -> Result<(Device, CudaDevice)> {
    let device = Device::new_cuda(0)?;
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!("new_cuda returns a CUDA device"),
    };
    Ok((device, dev))
}

/// Compare two KO twins byte for byte, and say something useful when they differ.
///
/// `assert_eq!` on the vectors themselves would print both operands in full — a 34 MiB twin is
/// tens of millions of numbers, and reading them answers nothing. What identifies the fault is
/// *which chunk* diverged: the row-group and k-block it fell in, whether the divergence starts
/// at a band boundary, and which plane it sits in. A divergence confined to `dm` is the
/// (scale, min) pair, i.e. float rounding in the observer; one in `ql` is the codes, i.e. a
/// layout or permutation fault. They have nothing to do with each other, and the byte offset
/// within the chunk is the only thing that separates them.
fn assert_ko_bytes_match(
    got: &[u8],
    want: &[u8],
    nrows: usize,
    ncols: usize,
    ko: GgmlDType,
    what: &str,
) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: twin byte length changed: {} vs {}",
        got.len(),
        want.len()
    );
    let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) else {
        return;
    };
    let chunk_bytes = ko_quant::ko_chunk_bytes(ko);
    let row_groups = nrows / 8;
    let k_blocks = ncols / 128;
    let chunk = i / chunk_bytes;
    let (k_blk, g) = (chunk / row_groups, chunk % row_groups);
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    let bad_chunks: std::collections::BTreeSet<usize> =
        bad.iter().map(|&i| i / chunk_bytes).collect();
    let bad_groups: std::collections::BTreeSet<usize> =
        bad_chunks.iter().map(|c| c % row_groups).collect();
    let in_dm = bad.iter().filter(|&&i| i % chunk_bytes >= 512).count();
    let max_delta = bad
        .iter()
        .map(|&i| got[i].abs_diff(want[i]))
        .max()
        .unwrap_or(0);
    panic!(
        "{what}: {} of {} bytes differ, in {} of {} chunks; {in_dm} of them in the dm \
         (scale,min) plane and {} in ql (codes). Largest byte delta {max_delta}.\n\
         first at byte {i} (chunk {chunk} = k_blk {k_blk}, row-group {g}, offset {} in \
         chunk): got {} want {}\nrow-groups touched: {:?}{}",
        bad.len(),
        got.len(),
        bad_chunks.len(),
        k_blocks * row_groups,
        bad.len() - in_dm,
        i % chunk_bytes,
        got[i],
        want[i],
        bad_groups.iter().take(16).collect::<Vec<_>>(),
        if bad_groups.len() > 16 { " …" } else { "" },
    );
}

/// **The repack's scratch is a band, not the tensor — and this is what says so.**
///
/// `repack_ko` is dequantize-then-requantize composed through an f32 buffer, and that buffer
/// used to be the whole tensor: 4,850 MiB for the 27B's `[248320, 5120]` head. Not merely large
/// but *permanent* — `dense_span` sized the span's `cuMemAddressReserve` smaller by exactly that
/// figure, and a reservation cannot grow, so a buffer alive for one tensor during load cost a
/// third of the card until the process exited. Repacking a row band at a time caps the
/// intermediate at `REPACK_BAND_BYTES` whatever the tensor's size.
///
/// So this measures free VRAM across the repack and requires the dip to stay near the twin.
/// The tensor is deliberately shaped so a whole-tensor f32 (256 MiB) dwarfs both the twin
/// (34 MiB) and the band (48 MiB) — a regression that reinstated the old buffer could not hide
/// inside the bound, and one that merely enlarged the band would have to grow it fivefold.
///
/// The second assertion is that the output did not change: byte-identical to the CPU codec
/// over the same dequantized source. Banding rearranges *when* each chunk is written and
/// scatters the results into place, which is exactly the kind of change that can produce a
/// correctly-sized, plausibly-valued, wrong tensor — so the comparison is on bytes.
#[test]
fn ko_repack_scratch_is_a_bounded_band() -> Result<()> {
    let (device, dev) = card()?;
    // Big enough that a whole-tensor f32 dwarfs both the source and the twin, so the bound
    // below is not competing with allocator granularity: 8192×8192 is 256 MiB of f32 against a
    // 37.7 MiB Q4_K source and a 35.7 MiB Q4_KO twin.
    let (nrows, ncols) = (8192usize, 8192usize);
    let n = nrows * ncols;
    let f32_bytes = n * 4;

    let w: Vec<f32> = (0..n).map(|i| ((i % 251) as f32 - 125.0) * 0.003).collect();
    let w_t = Tensor::from_vec(w, (nrows, ncols), &device)?;
    let src = QTensor::quantize(&w_t, GgmlDType::Q4_K)?;
    let shape = src.shape().clone();
    let storage = match src.storage() {
        QStorage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };

    let _alone = measuring_vram();
    dev.cuda_stream().synchronize().map_err(Error::wrap)?;
    let (free_before, _) = get_vram_info()?;
    let twin = storage.repack_ko(&shape, GgmlDType::Q4_KO)?;
    dev.cuda_stream().synchronize().map_err(Error::wrap)?;
    let (free_after, _) = get_vram_info()?;

    // What the device is *entitled* to hold across the repack: the twin it produced, plus the
    // f32 band and the KO band that produced it. The source was already resident before the
    // measurement, so it is not in the delta. Slack covers CUDA pool granularity, which rounds
    // allocations up generously.
    const SLACK: usize = 2 * cuda::REPACK_BAND_BYTES;
    let twin_bytes = cuda::ko_repacked_bytes(&shape, GgmlDType::Q4_KO)?;
    let used = free_before.saturating_sub(free_after);
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!(
        "repack VRAM delta {:.1} MiB | twin {:.1} | an on-device f32 would add {:.1}",
        mib(used),
        mib(twin_bytes),
        mib(f32_bytes),
    );
    assert!(
        used <= twin_bytes + SLACK,
        "repack held {:.1} MiB of VRAM; the twin is {:.1} MiB and the f32 intermediate would \
         be {:.1} MiB. The scratch is back on the card — and the span concedes that much \
         permanently, because `cuMemAddressReserve` sizes it once and cannot grow.",
        mib(used),
        mib(twin_bytes),
        mib(f32_bytes),
    );

    // And the twin is still what the CPU codec produces from the same dequantized source —
    // byte for byte. Moving where the intermediate lives must not move a single output bit,
    // and only a byte comparison says so; a size check would pass on any kernel at all.
    assert_eq!(twin.dtype(), GgmlDType::Q4_KO);
    let got: Vec<u8> = twin.data()?;
    let deq = storage.dequantize(n)?;
    let src_f32: Vec<f32> = dev
        .memcpy_dtov(deq.as_cuda_slice::<f32>()?)
        .map_err(Error::wrap)?;
    let want = ko_quant::quantize_ko(&src_f32, nrows, ncols, GgmlDType::Q4_KO);
    assert_ko_bytes_match(
        &got,
        &want,
        nrows,
        ncols,
        GgmlDType::Q4_KO,
        "the host-mapped intermediate changed the repack's output bytes",
    );
    Ok(())
}

/// **The float source takes the banded route too, and this is what says so.**
///
/// A source that arrives already float — an F16 `output.weight`, which real checkpoints carve
/// out (`…NEO-IMATRIX-MAX…` does) — had no `dtype_to_qtype` arm, so `repack_ko_into` could not
/// read it and `QMatMul::build` fell back to converting it whole: dequantize the tensor to f32,
/// copy it again inside `QTensor::quantize`'s `force_contiguous`, quantize *that* to an
/// intermediate `Q8_0`, and only then repack. Four whole-tensor buffers live at once. On a
/// `[248320, 4096]` head it was 10,730 MiB to produce a twin the banded route builds inside a
/// 48 MiB band, and it OOMed a 24 GiB card with 14 GiB free — reported only as
/// `CUDA_ERROR_OUT_OF_MEMORY`, naming no tensor and no size.
///
/// So this asserts the same two properties as the test above, against the path that did not
/// have them: the VRAM dip stays near the twin, and the bytes match the CPU codec exactly.
/// The second is not a formality — widening with a cast where the quantized path dequantizes is
/// a different kernel writing the same buffer, which is exactly how a correctly-sized,
/// plausibly-valued, wrong tensor gets made.
#[test]
fn a_float_source_repacks_through_the_same_bounded_band() -> Result<()> {
    let (device, dev) = card()?;
    // Same shape as the quantized sibling, so the two are directly comparable: 8192×8192 is
    // 256 MiB of f32 — five times the band — against a 128 MiB F16 source and a 68 MiB twin.
    let (nrows, ncols) = (8192usize, 8192usize);
    let n = nrows * ncols;
    let f32_bytes = n * 4;

    let w: Vec<f32> = (0..n).map(|i| ((i % 251) as f32 - 125.0) * 0.003).collect();
    // An F16 tensor, quantized to F16 — i.e. stored as-is, which is what a checkpoint that
    // carves a tensor out at higher precision hands the loader.
    let w_t = Tensor::from_vec(w, (nrows, ncols), &device)?.to_dtype(DType::F16)?;
    let src = QTensor::quantize(&w_t, GgmlDType::F16)?;
    assert_eq!(src.dtype(), GgmlDType::F16);
    // The routing predicate must admit it, or the loader never reaches this path at all.
    assert!(
        cuda::repackable_to_ko(GgmlDType::F16),
        "a float source must route to the banded repack"
    );
    assert!(
        !cuda::gemx_repacking_supported(GgmlDType::F16),
        "…and it must do so without claiming a GEMX kernel it does not have"
    );

    let shape = src.shape().clone();
    let storage = match src.storage() {
        QStorage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };

    let _alone = measuring_vram();
    dev.cuda_stream().synchronize().map_err(Error::wrap)?;
    let (free_before, _) = get_vram_info()?;
    let twin = storage.repack_ko(&shape, GgmlDType::Q8_KO)?;
    dev.cuda_stream().synchronize().map_err(Error::wrap)?;
    let (free_after, _) = get_vram_info()?;

    // The entitlement is computed, not guessed: the twin, plus exactly the two bands
    // `repack_ko_into` sizes for this shape. A borrowed constant would be wrong here — the
    // sibling test's twin is Q4_KO, and a Q8_KO twin's KO band is proportionally larger — and
    // loosening a constant until a test passes is how a bound stops meaning anything.
    let twin_bytes = cuda::ko_repacked_bytes(&shape, GgmlDType::Q8_KO)?;
    let band_rows = ((cuda::REPACK_BAND_BYTES / (ncols * 4)) / 8)
        .max(1)
        .min(nrows / 8)
        * 8;
    let f32_band = band_rows * ncols * 4;
    let ko_band = band_rows / 8 * (ncols / 128) * ko_quant::ko_chunk_bytes(GgmlDType::Q8_KO);
    let used = free_before.saturating_sub(free_after);
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!(
        "float repack VRAM delta {:.1} MiB | twin {:.1} + f32 band {:.1} + ko band {:.1} | \
         the old route: source {:.1} + f32 {:.1} + its copy {:.1} + Q8_0 intermediate",
        mib(used),
        mib(twin_bytes),
        mib(f32_band),
        mib(ko_band),
        mib(n * 2),
        mib(f32_bytes),
        mib(f32_bytes),
    );

    // **The threshold is deliberately loose, and the looseness is the honest part.**
    //
    // A tight bound here measures the CUDA pool's rounding, not the code: the same repack
    // reported 192 MiB and 224 MiB on consecutive runs against an arithmetic figure of 126.
    // Chasing that with a constant tuned until it passed would produce a test that fails on
    // somebody else's driver and tells them nothing.
    //
    // What the test is actually for is one question — did the whole-tensor conversion come
    // back — and the two answers are far apart. Banded is band-scale: the twin plus two bands,
    // ~200 MiB here. The old route materialised the tensor four times over (source 128 + f32
    // 256 + `force_contiguous`'s copy 256 + a Q8_0 intermediate 68) before it repacked
    // anything: ~700 MiB. A line drawn at the twin plus one whole-tensor f32 sits three
    // hundred MiB clear of one and two hundred clear of the other.
    let whole_tensor_scale = twin_bytes + f32_bytes;
    assert!(
        used < whole_tensor_scale,
        "repacking a float source held {:.1} MiB — whole-tensor scale. The conversion is no \
         longer banded, which is the route that OOMed a 24 GiB card with 14 GiB free.",
        mib(used),
    );

    // Byte-identical to the CPU codec over the same values. The source is F16, so the
    // reference dequantizes it the same way the band does.
    assert_eq!(twin.dtype(), GgmlDType::Q8_KO);
    let got: Vec<u8> = twin.data()?;
    let deq = storage.dequantize(n)?;
    let src_f32: Vec<f32> = dev
        .memcpy_dtov(deq.as_cuda_slice::<f32>()?)
        .map_err(Error::wrap)?;
    let want = ko_quant::quantize_ko(&src_f32, nrows, ncols, GgmlDType::Q8_KO);
    assert_ko_bytes_match(
        &got,
        &want,
        nrows,
        ncols,
        GgmlDType::Q8_KO,
        "the widening cast changed the repack's output bytes",
    );
    Ok(())
}

/// **The source never reaches the device whole, and the bytes are the same anyway.**
///
/// Banding the f32 intermediate left one whole-tensor allocation standing: getting the source
/// onto the card so `repack_ko_into` could read it. For a `[248320, 4096]` BF16 `output.weight`
/// that is 1,940 MiB which exists only to be read once, in order, and dropped — and it is what
/// finally OOMed a 24 GiB card with 14 GiB free, because the load budget's margin over it was a
/// few tens of MiB and allocator granularity ate them.
///
/// `repack_ko_from_host` reads the mapping a band at a time instead, so the device peak is
/// `staging + f32 + ko` bands and does not scale with the tensor at all. Two assertions: the
/// device delta stays band-scale, and the twin is byte-identical to what the device-sourced
/// path produces. The second is the one that matters most — staging changes *when* each row
/// reaches the kernel, which is exactly how a correctly-sized, plausibly-valued, wrong tensor
/// gets made.
#[test]
fn a_host_banded_repack_matches_the_device_one_without_materialising_the_source() -> Result<()> {
    let (device, dev) = card()?;
    // Big enough that the source dwarfs the bands: 16384×4096 BF16 is 128 MiB of source
    // against a 48 MiB f32 band, so a regression that materialised it could not hide inside
    // the threshold.
    let (nrows, ncols) = (16384usize, 4096usize);
    let n = nrows * ncols;
    let shape = Shape::from((nrows, ncols));

    let w: Vec<f32> = (0..n)
        .map(|i| ((i % 397) as f32 - 198.0) * 0.00390625)
        .collect();
    let w_t = Tensor::from_vec(w, (nrows, ncols), &device)?.to_dtype(DType::BF16)?;
    let src = QTensor::quantize(&w_t, GgmlDType::BF16)?;
    // The tensor's bytes exactly as GGUF lays them out — which is what the loader hands this
    // function from the memory-mapped checkpoint.
    let host: Vec<u8> = src.data()?.to_vec();
    let src_bytes = n * 2;
    assert_eq!(host.len(), src_bytes, "BF16 source is 2 bytes an element");

    // The reference: the device-sourced path, whose bytes are already locked by
    // `ko_repack_scratch_is_a_bounded_band`.
    let want = {
        let storage = match src.storage() {
            QStorage::Cuda(s) => s,
            _ => panic!("expected CUDA storage"),
        };
        storage.repack_ko(&shape, GgmlDType::Q8_KO)?.data()?
    };
    // Drop the device-resident source before measuring, so the delta below is the host path's
    // own and not this reference's.
    drop(src);
    let _alone = measuring_vram();
    device.synchronize()?;

    let (free_before, _) = get_vram_info()?;
    let twin =
        cuda::repack_ko_from_host(&dev, &host, &shape, GgmlDType::BF16, GgmlDType::Q8_KO, None)?;
    device.synchronize()?;
    let (free_after, _) = get_vram_info()?;

    let twin_bytes = cuda::ko_repacked_bytes(&shape, GgmlDType::Q8_KO)?;
    let used = free_before.saturating_sub(free_after);
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!(
        "host-banded repack VRAM delta {:.1} MiB | twin {:.1} | the source it did NOT upload \
         {:.1}",
        mib(used),
        mib(twin_bytes),
        mib(src_bytes),
    );
    // The twin, plus four bands' worth of slack for the three band buffers and pool
    // granularity. Loose for the reason the sibling test records — the same repack measured
    // 96, 192 and 224 MiB on consecutive runs — but decisive: uploading the source would add
    // 128 MiB and put it over.
    let entitled = twin_bytes + 4 * cuda::REPACK_BAND_BYTES;
    assert!(
        used <= entitled,
        "host-banded repack held {:.1} MiB against {:.1}; the source is being materialised on \
         the device again, which is the allocation that OOMed a 24 GiB card",
        mib(used),
        mib(entitled),
    );

    // And byte-identical to the device-sourced repack. Staging rows through a small buffer
    // must not move a single output bit.
    let got: Vec<u8> = twin.data()?;
    assert_ko_bytes_match(
        &got,
        &want,
        nrows,
        ncols,
        GgmlDType::Q8_KO,
        "the host-banded read changed the repack's output bytes",
    );
    Ok(())
}
