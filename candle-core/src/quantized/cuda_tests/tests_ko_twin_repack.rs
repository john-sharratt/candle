//! Repacking a weight into its KO twin on the load path.
//!
//! The repack writes into a destination inside the device reservation, so a
//! repack that fails part-way must not free the memory it was lent. The test
//! provokes a failure after the lease and checks that a recycling request still
//! sees the sentinel. The band size is asserted to be independent of tensor
//! size, which guards against a whole-tensor f32 buffer held for the life of the
//! process. Float sources are reconstructed on the CPU and matched across band
//! boundaries, with Q8_0 as the control.

use super::*;

/// A repack that fails part-way must not free the destination it was lent.
///
/// **The highest-consequence shape of bug this file can have.** `repack_ko_into` takes an
/// optional `dst`, and on the load path that is an address inside the device reservation — the
/// dense block. `dest_slice` wraps it with `upgrade_device_ptr`, which produces a `CudaSlice`
/// that *owns* its pointer, so anything that drops that slice calls `cuMemFreeAsync` on memory
/// the caller does not own. Every weight placed after it would then be sitting on ground the
/// pool believes is free.
///
/// For a long time that was safe by accident: there was no `?` between `dest_slice` and the
/// constructor, so the slice could not be dropped. Banding the repack added two fallible steps
/// inside that window, and nothing failed — because nothing exercised the window. This does.
///
/// The provocation is a source `dequantize_f32_into` refuses on the first band, which is
/// *after* `dest_slice` and before the destination is written. The
/// detector is recycling pressure — a freed block goes back to the CUDA pool, and a same-sized
/// request immediately afterwards is very likely to be handed it, so a lease that was wrongly
/// freed shows up as a clobbered sentinel rather than as a silent success.
#[test]
pub(super) fn a_failed_repack_does_not_free_its_leased_destination() -> Result<()> {
    use crate::cuda_backend::wave_provenance::LeaseOrigin;
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let stream = dev.cuda_stream();
    let (nrows, ncols) = (2048usize, 2048usize);
    let shape = crate::Shape::from((nrows, ncols));

    // Stand-in for the dense block: memory this test owns, which the repack is lent and must
    // return untouched. Big enough that the allocator will recycle it on a matching request.
    let dst_len = crate::quantized::cuda::ko_repacked_bytes(&shape, GgmlDType::Q4_KO)?;
    let sentinel = vec![0xA5u8; dst_len];
    let mut lease = unsafe { dev.alloc::<u8>(dst_len)? };
    dev.memcpy_htod(&sentinel, &mut lease)?;
    let lease_ptr = {
        let (p, _g) = lease.device_ptr(&stream);
        p
    };

    // A source with a tileable shape, so it reaches the band loop, and no device dequant
    // kernel, so the loop's first dequantize refuses.
    //
    // **This was F16, and F16 stopped working as a provocation** when the float formats gained
    // a widening arm in `dequantize_f32_into` — they now repack like any block format, which
    // is the whole point of that change. `P2` is a pure arena-routing format with no matmul
    // path at all, so it is refused for a reason that will not be engineered away: a weight is
    // never a palette index.
    let w = crate::Tensor::zeros((nrows, ncols), crate::DType::F32, &device)?;
    let src = crate::quantized::QTensor::quantize(&w, GgmlDType::P2)?;
    let storage = match src.storage() {
        crate::quantized::QStorage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };

    let err = storage
        .repack_ko_into(
            &shape,
            GgmlDType::Q4_KO,
            Some((lease_ptr, LeaseOrigin::Foreign)),
        )
        .expect_err("a P2 source has no QType and must not repack");
    let msg = err.to_string();
    assert!(
        msg.contains("has no device dequant kernel"),
        "expected the failure to come from the band loop's dequantize, got: {msg}"
    );

    // Recycling pressure. If the lease was freed, this request is the one that takes it.
    let clobber = vec![0x5Au8; dst_len];
    let mut probe = unsafe { dev.alloc::<u8>(dst_len)? };
    dev.memcpy_htod(&clobber, &mut probe)?;
    dev.synchronize()?;

    let back: Vec<u8> = dev
        .memcpy_dtov(&lease.slice(..))
        .map_err(crate::Error::wrap)?;
    assert_eq!(
        back, sentinel,
        "the failed repack freed the destination it was lent — on the load path that address \
         is inside the device reservation, and the pool has just handed it to someone else"
    );
    Ok(())
}

/// **The band is bounded whatever the tensor — asserted without a device.**
///
/// The history this protects: `repack_ko` is dequantize-then-requantize composed through an
/// f32 buffer, and that buffer used to be the whole tensor — 4,850 MiB for the 27B's
/// `[248320, 5120]` head. Not merely large but *permanent*: `dense_span` sized the span's
/// `cuMemAddressReserve` smaller by exactly that figure, and a reservation cannot grow, so a
/// buffer alive for one tensor during load cost a third of the card until the process exited.
/// Repacking a row band at a time caps the intermediate at `REPACK_BAND_BYTES` whatever the
/// tensor's size.
///
/// `candle-core/tests/vram_bounds.rs` measures that end to end, through free VRAM. This one
/// computes the same number the allocation uses, so it needs no device and no quiet card —
/// it runs anywhere, every time, which is what makes it the primary guard rather than the
/// backstop.
///
/// The regression it would catch is the one that motivated the banding: a
/// whole-tensor f32, which at 8192×8192 is 256 MiB and grows with the model.
#[test]
pub(super) fn ko_repack_band_is_bounded_regardless_of_tensor_size() {
    use crate::quantized::cuda::{repack_band_bytes, REPACK_BAND_BYTES};
    // Square, tall-thin and short-wide, across three orders of magnitude.
    for (nrows, ncols) in [
        (8192usize, 8192usize),
        (65536, 8192),
        (1024, 1024),
        (8, 32768),
        (131072, 256),
    ] {
        let band = repack_band_bytes(nrows, ncols);
        let whole = nrows * ncols * std::mem::size_of::<f32>();
        assert!(
            band <= REPACK_BAND_BYTES.max(ncols * 8 * std::mem::size_of::<f32>()),
            "{nrows}x{ncols}: band {band} B exceeds the {REPACK_BAND_BYTES} B bound \
             (one row-group is the floor, for a tensor too wide to fit even that)"
        );
        assert!(
            band <= whole,
            "{nrows}x{ncols}: a band may never exceed the whole tensor it is banding"
        );
    }
    // And it does NOT grow with row count: the whole point of banding.
    assert_eq!(
        repack_band_bytes(8192, 8192),
        repack_band_bytes(65536, 8192),
        "the band must not scale with rows — that is what makes it a band"
    );
}

// The three repack-band tests that used to sit here now live in
// `candle-core/tests/vram_bounds.rs`. They bracket an operation with
// `get_vram_info`, which reports free memory for the whole card, so every
// allocation the other tests in this binary made inside that window landed in
// the delta — 1,226 MiB measured against a 258 MiB entitlement, failing at
// random while the same three passed under `--test-threads=1`. An integration
// test is its own process, which is the only place that measurement means
// anything.

/// **The host-banded read must match the device one for every source it accepts**, not just
/// the float ones that motivated it.
///
/// The loader routes *every* repackable projection through `repack_ko_from_host` when it has
/// the mapping, so a bug that only shows on block-quantized sources is a bug in most of the
/// model — and the first version of this file tested BF16 alone, which is how a whole
/// checkpoint's worth of weights went unchecked.
#[test]
pub(super) fn a_host_banded_repack_matches_the_device_one_for_quantized_sources_too() -> Result<()>
{
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    // Shapes and formats a real checkpoint actually carries, including a non-square
    // projection and a source whose blocks are 256 wide.
    for (nrows, ncols, dtype, ko) in [
        (4096usize, 4096usize, GgmlDType::Q6_K, GgmlDType::Q6_KO),
        (12288, 4096, GgmlDType::Q6_K, GgmlDType::Q6_KO),
        (4096, 12288, GgmlDType::Q4_K, GgmlDType::Q5_KO),
        (2048, 4096, GgmlDType::Q8_0, GgmlDType::Q8_KO),
    ] {
        let n = nrows * ncols;
        let w: Vec<f32> = (0..n)
            .map(|i| ((i % 313) as f32 - 156.0) * 0.00390625)
            .collect();
        let w_t = crate::Tensor::from_vec(w, (nrows, ncols), &device)?;
        let src = crate::quantized::QTensor::quantize(&w_t, dtype)?;
        let shape = src.shape().clone();
        let host: Vec<u8> = src.data()?.to_vec();

        let want = {
            let storage = match src.storage() {
                crate::quantized::QStorage::Cuda(s) => s,
                _ => panic!("expected CUDA storage"),
            };
            storage.repack_ko(&shape, ko)?.data()?
        };
        let got: Vec<u8> =
            crate::quantized::cuda::repack_ko_from_host(&dev, &host, &shape, dtype, ko, None)?
                .data()?;

        assert_eq!(
            got.len(),
            want.len(),
            "{dtype:?} [{nrows}, {ncols}] → {ko:?}: twin length differs"
        );
        assert_eq!(
            got, want,
            "{dtype:?} [{nrows}, {ncols}] → {ko:?}: the host-banded read produced different \
             bytes from the device one. Every projection in a mapped load takes this path."
        );
    }
    Ok(())
}

/// **A float source's KO twin must hold the source's numbers, not merely agree with itself.**
///
/// Every other test around this path compares the host-banded read against the device one. That
/// catches a *disagreement* and is blind to the case where both routes are wrong together —
/// which is the case that matters here, because they share `dequantize_f32_into` and the float
/// arms of it are the newest code in the file.
///
/// So this one is semantic: quantize to a KO twin, pull the bytes back, and reconstruct them
/// with `ko_quant::dequant_ko` — a CPU reference implementation that shares nothing with the
/// CUDA path. If the widen reads at the wrong stride, swaps the halves of a bf16 word, or
/// widens the wrong element count, the reconstruction is wrong and no amount of host/device
/// agreement hides it.
///
/// `Q8_0` is here as the control. It is what the reference checkpoint carries for `ssm_out`
/// and is known-good in production, so a failure on the float rows *with* `Q8_0` passing is a
/// float-widen bug, and a failure on all four is a bug in the repack proper.
///
/// # Why this path is worth its own test
///
/// A float weight had no KO twin at all until recently: `to_ko` bailed, and the loader left
/// float projections on the FP path (see `quantized_qwen35.rs`, which records that being the
/// prime suspect behind the 0.8B's KV factors sitting ~3× tighter than its siblings'). Giving
/// floats a twin routed a whole class of weight through the int8 kernels for the first time,
/// and the checkpoints that carry them — third-party conversions with BF16 `ssm_*` tensors
/// where the reference has `F32`/`Q8_0` — are exactly the ones that generate word salad.
#[test]
pub(super) fn a_float_sources_ko_twin_reconstructs_its_source_values() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    // A DeltaNet output projection's proportions, small enough to reconstruct on the CPU.
    let (nrows, ncols) = (512usize, 1024usize);
    let n = nrows * ncols;
    // Deliberately not symmetric about zero and not a round power of two: an affine twin with
    // a sign error or a dropped `min` still reconstructs symmetric data passably.
    let w: Vec<f32> = (0..n)
        .map(|i| ((i % 397) as f32 - 120.0) * 0.011_718_75)
        .collect();
    let w_t = crate::Tensor::from_vec(w.clone(), (nrows, ncols), &device)?;

    for dtype in [
        GgmlDType::Q8_0,
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
    ] {
        let cast = match dtype {
            GgmlDType::F16 => w_t.to_dtype(crate::DType::F16)?,
            GgmlDType::BF16 => w_t.to_dtype(crate::DType::BF16)?,
            _ => w_t.clone(),
        };
        let src = crate::quantized::QTensor::quantize(&cast, dtype)?;
        assert_eq!(
            src.dtype(),
            dtype,
            "{dtype:?}: source did not keep its format"
        );
        let shape = src.shape().clone();
        let host: Vec<u8> = src.data()?.to_vec();

        // **Both routes, because the loader picks between them on a condition unrelated to
        // correctness.** `host_banded` takes the mapping when there is one; `QMatMul::build`
        // repacks the device tensor when there is not. A weight is equally live either way, so
        // a float bug in one of them is a float bug in half the loads — and testing only the
        // host route is how this test first passed while the model still generated garbage.
        let device_twin = {
            let storage = match src.storage() {
                crate::quantized::QStorage::Cuda(s) => s,
                _ => panic!("expected CUDA storage"),
            };
            storage.repack_ko(&shape, GgmlDType::Q8_KO)?.data()?
        };
        let host_twin: Vec<u8> = crate::quantized::cuda::repack_ko_from_host(
            &dev,
            &host,
            &shape,
            dtype,
            GgmlDType::Q8_KO,
            None,
        )?
        .data()?;

        // The source's own loss plus Q8's. BF16 keeps 8 mantissa bits, so it dominates its row;
        // the others are bounded by the Q8 grid over this data's range.
        let tol = match dtype {
            GgmlDType::BF16 => 0.12,
            GgmlDType::F16 => 0.05,
            _ => 0.05,
        };
        for (route, bytes) in [("host-banded", &host_twin), ("device", &device_twin)] {
            let back =
                crate::quantized::ko_quant::dequant_ko(bytes, nrows, ncols, GgmlDType::Q8_KO);
            assert_eq!(
                back.len(),
                n,
                "{dtype:?} via {route}: reconstruction has the wrong length"
            );

            let mut worst = 0f32;
            let mut worst_at = 0usize;
            for (i, (g, s)) in back.iter().zip(w.iter()).enumerate() {
                let d = (g - s).abs();
                if d > worst {
                    worst = d;
                    worst_at = i;
                }
            }
            assert!(
                worst <= tol,
                "{dtype:?} → Q8_KO via {route}: reconstruction is wrong. Worst element \
                 {worst_at}: twin says {}, source says {} (off by {worst}, tolerance {tol}). \
                 The twin does not hold this weight's numbers, so every matmul against it is \
                 wrong.",
                back[worst_at],
                w[worst_at]
            );
        }

        assert_eq!(
            host_twin, device_twin,
            "{dtype:?} → Q8_KO: the two routes produced different bytes, so which one a load \
             happens to take changes the weights"
        );
    }
    Ok(())
}

/// The widening arm itself, in isolation and against every float format.
///
/// `dequantize_f32_into` is what makes a float source readable by the banded repack, and it is
/// a *cast* where the quantized arms are dequant kernels — F32 is not even that, it is a copy.
/// Three different routines behind one signature is three chances to widen the wrong number of
/// elements or read at the wrong stride, and the band loop hides an off-by-one inside the
/// second band where a single-band test would never look. So the tensor is deliberately larger
/// than one band.
#[test]
pub(super) fn the_float_widen_matches_its_source_across_band_boundaries() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    // 24 MiB of f32 — several bands at 48 MiB per band's worth of f32? No: bands are sized in
    // f32 bytes, so 32 M elements is 128 MiB of f32 and crosses the boundary twice.
    let n = 32 * 1024 * 1024usize;
    let (nrows, ncols) = (n / 4096, 4096);
    let w: Vec<f32> = (0..n)
        .map(|i| ((i % 509) as f32 - 254.0) * 0.0078125)
        .collect();
    let w_t = crate::Tensor::from_vec(w.clone(), (nrows, ncols), &device)?;

    for dtype in [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16] {
        let cast = match dtype {
            GgmlDType::F16 => w_t.to_dtype(crate::DType::F16)?,
            GgmlDType::BF16 => w_t.to_dtype(crate::DType::BF16)?,
            _ => w_t.clone(),
        };
        let q = crate::quantized::QTensor::quantize(&cast, dtype)?;
        assert_eq!(q.dtype(), dtype);
        let back = q.dequantize(&device)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(back.len(), n, "{dtype:?} widened the wrong element count");

        // The tolerance is the format's own, not the kernel's: F32 is exact, F16 and BF16 lose
        // mantissa bits at conversion and nothing downstream can recover them. A single
        // tolerance for all three would either fail F32's exactness or hide a BF16 bug.
        let tol = match dtype {
            GgmlDType::F32 => 0.0,
            GgmlDType::F16 => 1e-3,
            _ => 1e-2,
        };
        for (i, (g, s)) in back.iter().zip(w.iter()).enumerate() {
            assert!(
                (g - s).abs() <= tol,
                "{dtype:?} element {i}: widened {g}, source {s}"
            );
        }
    }
    Ok(())
}

/// Every affine KO twin has a fused-QKV format code, and every code is in the kernel's range.
///
/// **The failure mode here is silence, which is why it needs its own test.** `ko_fmt_code`
/// turns a dtype into the `fmt` index the segmented kernel switches on, and that switch's
/// `default` arm cannot fault — it falls through, leaving that segment's output columns
/// unwritten. A caller sees a completed launch and plausible numbers for the other two
/// segments. So an unmapped dtype must be refused in Rust, before a code is ever produced, and
/// a mapped one must land inside `QKV_FMT_COUNT`.
///
/// `MXFP4_KO` is the one KO format deliberately without a code: it is not affine, its per-sub
/// scale fold is a different kernel, and the fused-QKV path never carries it.
#[test]
pub(super) fn every_affine_ko_twin_has_a_qkv_format_code() -> Result<()> {
    // Must equal `QKV_FMT_COUNT` in `candle-kernels/src/quantized/impl/qkv_segmented_f32.cu`,
    // which bounds both the `kuni` dispatch table and the launcher's range check.
    const QKV_FMT_COUNT: i32 = 6;

    let mut codes = Vec::new();
    for &dtype in crate::quantized::ko_quant::ALL_KO_DTYPES {
        let got = super::super::ko_fmt_code(dtype);
        if dtype == GgmlDType::MXFP4_KO {
            assert!(
                got.is_err(),
                "MXFP4_KO is not affine and the fused-QKV path does not carry it"
            );
            continue;
        }
        let code = got.map_err(|e| {
            crate::Error::Msg(format!(
                "{dtype:?} has no fused-QKV format code, so a q/k/v projection at that width \
                 fails the whole model: {e}"
            ))
        })?;
        assert!(
            (0..QKV_FMT_COUNT).contains(&code),
            "{dtype:?} maps to fmt {code}, outside the kernel's 0..{QKV_FMT_COUNT} switch — \
             the default arm leaves that segment's columns unwritten"
        );
        codes.push(code);
    }

    // Distinct codes: two dtypes sharing one index is a segment silently multiplied by another
    // format's weights, which produces plausible numbers rather than an error.
    let n = codes.len();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(
        n,
        codes.len(),
        "two KO dtypes share a fused-QKV format code"
    );

    // A non-KO dtype must be refused rather than coerced onto some default index.
    for d in [GgmlDType::Q4_K, GgmlDType::Q8_0, GgmlDType::F16] {
        assert!(
            super::super::ko_fmt_code(d).is_err(),
            "{d:?} is not a KO twin and must not receive a format code"
        );
    }
    Ok(())
}

/// Every KO twin loads from GGUF bytes, and the loaded tensor is byte-identical to what was
/// written.
///
/// **The gap this closes was a hard refusal, not a slow path.** `qtensor_from_ggml`'s KO arm
/// listed the formats by name and its fallback is `bail!`, so a prepared GGUF carrying a format
/// missing from that list could not be opened at all — and `Q2_KO` was missing from the day it
/// was added, because nothing here swept the set. Listing the dtypes in a test is no better
/// than listing them in the loader unless the list is derived, so this sweeps every `is_ko()`
/// dtype and skips only what it cannot synthesise.
///
/// The byte comparison matters as much as the acceptance: this path is the one that must copy
/// straight to VRAM with **no** `MATRIX_ROW_PADDING` and no reinterpret, so a tensor that
/// loaded but grew a padding tail would still be wrong for the KO matmul, which reads chunks.
#[test]
pub(super) fn every_ko_twin_loads_from_ggml_bytes() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (nrows, ncols) = (64usize, 256usize);
    let w: Vec<f32> = (0..nrows * ncols)
        .map(|i| ((i % 251) as f32 - 125.0) * 0.004)
        .collect();

    let mut covered = 0usize;
    for &dtype in crate::quantized::ko_quant::ALL_KO_DTYPES {
        // MXFP4_KO is not affine — `quantize_ko` cannot produce it, and its GGUF form comes
        // from `mxfp4_native_to_ko_gpu_chunk` over native MXFP4 blocks instead.
        let bytes = if dtype == GgmlDType::MXFP4_KO {
            crate::quantized::ko_quant::mxfp4_ko_to_gpu_chunk(
                &crate::quantized::ko_quant::quantize_mxfp4_ko(&w, nrows, ncols),
                nrows,
                ncols,
            )
        } else {
            crate::quantized::ko_quant::quantize_ko(&w, nrows, ncols, dtype)
        };

        // What the GGUF header would compute for this tensor must equal the bytes we hold, or
        // the loader slices the wrong length out of the file.
        let expect = nrows * ncols / dtype.block_size() * dtype.type_size();
        assert_eq!(
            bytes.len(),
            expect,
            "{dtype:?}: type_size accounting disagrees with the emitted chunk"
        );

        let qt = crate::quantized::ggml_file::qtensor_from_ggml(
            dtype,
            &bytes,
            vec![nrows, ncols],
            &device,
        )
        .map_err(|e| crate::Error::Msg(format!("{dtype:?} failed to load from GGUF bytes: {e}")))?;
        assert_eq!(qt.dtype(), dtype, "{dtype:?} loaded as another dtype");
        assert_eq!(
            qt.storage_size_in_bytes(),
            bytes.len(),
            "{dtype:?}: the KO load path must not pad — the matmul reads whole chunks"
        );
        let back = qt.data()?;
        assert_eq!(
            &back[..],
            &bytes[..],
            "{dtype:?} round trip is not byte-exact"
        );
        covered += 1;
    }
    assert_eq!(
        covered,
        crate::quantized::ko_quant::ALL_KO_DTYPES.len(),
        "every KO dtype must be exercised"
    );
    Ok(())
}

/// Repro guard for the offline-KO GGUF load path: a Q8_KO weight built from CPU
/// `quantize_ko` bytes uploaded via `load_repacked` (what `qtensor_from_ggml` now does for a
/// pre-KO GGUF) must forward-via-int8 identically to the same weight repacked on-GPU by
/// `repack_for_optimization` (the Cat2 at-load path). Both feed the SAME Q8_0-dequantized f32,
/// so any divergence (or crash) isolates the offline-load path from the engine.
#[test]
pub(super) fn ko_offline_load_forward_matches_gpu_repack() -> Result<()> {
    use crate::quantized::ko_quant::quantize_ko;
    use crate::quantized::{QMatMul, QTensor};
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (nrows, ncols, m) = (256usize, 512usize, 32usize); // out, in, batch
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let w_t = crate::Tensor::from_vec(wf32, (nrows, ncols), &device)?;
    let act_t = crate::Tensor::from_vec(act, (m, ncols), &device)?;

    // Path A (known-good, Cat2): Q8_0 → GPU repack_for_optimization → forward_via_int8.
    let q8 = QMatMul::from_qtensor(QTensor::quantize(&w_t, GgmlDType::Q8_0)?)?;
    let opt_a = q8.repack_for_optimization(Int8Mode::Performance)?;
    let y_a: Vec<f32> = opt_a
        .forward_via_int8(&act_t, Int8Mode::Performance, act_t.dtype(), SumScale::Raw)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Path B (offline emulation): dequant the SAME Q8_0 → CPU quantize_ko → load_repacked → wrap.
    let w_q8_deq = QTensor::quantize(&w_t, GgmlDType::Q8_0)?.dequantize(&device)?;
    let w_q8_vec = w_q8_deq.flatten_all()?.to_vec1::<f32>()?;
    let ko_bytes = quantize_ko(&w_q8_vec, nrows, ncols, GgmlDType::Q8_KO);
    let storage_b = crate::quantized::cuda::load_repacked(&dev, &ko_bytes, GgmlDType::Q8_KO)?;
    let qt_b = QTensor::new(storage_b, vec![nrows, ncols])?;
    let mm_b = QMatMul::from_qtensor(qt_b)?;
    let y_b: Vec<f32> = mm_b
        .forward_via_int8(&act_t, Int8Mode::Performance, act_t.dtype(), SumScale::Raw)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let rel = rel_l2(&y_a, &y_b);
    println!("offline-KO-load vs GPU-repack rel_l2 = {rel:.3e}");
    assert!(
        rel < 1e-3,
        "offline-KO-load path diverged from GPU repack: rel_l2={rel}"
    );
    Ok(())
}
