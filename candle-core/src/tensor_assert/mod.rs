//! Zero-sync finiteness assertions over tensors and quantized tensors.
//!
//! `x.assert("attn.qkv_in")` folds NaN count, Inf count and the finite min/max
//! of `x` into a preallocated device slot and returns immediately. It does not
//! synchronise, does not allocate, does not convert dtype, and does not copy the
//! tensor anywhere — the kernel reads the data where it already lives, through
//! its own offset and strides. The whole cost is one launch.
//!
//! That is what makes it usable on a hot path, and it is not an incidental
//! property. An instrument that synchronises or allocates inside the region it
//! observes is not an instrument, it is a change to the program: it reorders
//! launches against every asynchronous stream in flight and moves the arena
//! layout that the code under observation allocates from. A probe built the
//! obvious way — `sum_all().to_scalar()` at each checkpoint — does both, and
//! will report a clean run on a program that reliably fails without it.
//!
//! # Reading the results
//!
//! Nothing is read back until [`drain`] is called, which belongs at a
//! synchronisation the caller already performs. Because of that, host call order
//! cannot tell you which assert went bad first — so the kernel stamps each slot
//! with a ticket from one global counter the first time it goes bad, and
//! [`drain`] sorts by it. The first entry of that report is the answer to
//! "where did this originate"; everything after it is downstream.
//!
//! # What this cannot tell you, and what to reach for instead
//!
//! An assert answers "is this tensor finite", which is a narrower question than
//! it looks. Three limits worth knowing before trusting a report:
//!
//! * **Quantization erases a NaN.** `amax.max(x.abs())` returns the non-NaN
//!   operand, so a NaN never reaches a Q8_0 scale and `round(NaN) as i8`
//!   saturates to 0. A clean report *after* an int8 quantize therefore does not
//!   clear the values that went into it. An **Inf** does the opposite: it drives
//!   the scale to inf and turns its whole 32-element block to NaN. Both are
//!   pinned by tests in `candle-core/tests/tensor_assert_tests.rs`.
//! * **A finite value can still be wrong.** The fault this harness was built for
//!   presented as finite, plausibly-shaped expert weights that were simply
//!   another tenant's activations. Finiteness is necessary, not sufficient —
//!   when the operands and weights all pass and the output is still wrong,
//!   suspect memory, not arithmetic (CLAUDE.md hot-path invariant 7).
//! * **The order of the report is the kernel's, not the host's.** Nothing is
//!   read back until the drain, so host call order says nothing about which site
//!   went bad first. The `seq` ticket does; sort by it.
//!
//! For capturing *why* rather than *where*, `check_now` is the synchronous form
//! — it must sit in the same pass as the operands, because the next pass through
//! a site carries different ones and would faithfully record a call that did not
//! fail.
//!
//! # Weights
//!
//! Use [`LiveQTensor::assert_once`] for weights. A weight does not change
//! between forwards, so re-reading it every layer of every wave is exactly the
//! bandwidth perturbation this design exists to avoid. Quantized asserts also
//! run through the dequant kernels on the default stream and stage through a
//! grow-to-fit scratch, so they belong at load time or an epoch boundary — not
//! inside a wave.

pub mod callback;
pub mod drain;
pub mod dump;
pub mod names;
pub mod scratch;
pub mod slots;

use crate::cuda_backend::DeviceId;
use crate::quantized::{GgmlDType, LiveQTensor, QStorage};
use crate::{DType, Device, LiveTensor, Result, Storage};
use candle_kernels::simple::tensor_assert::{run_tensor_assert, AssertDType, MAX_DIMS};

pub use callback::on_bad;
pub use candle_kernels::simple::tensor_assert::AssertDType as RawDType;
pub use drain::{drain, find, report, Finding};
// `check_now` is defined below; re-exported here so the site-level API reads as
// one surface.
pub use dump::{Dump, Replay};
pub use scratch::QTYPE_Q8A128V;
pub use names::{interned_site as interned, rearm_once, site};
pub use slots::AssertSlot;

fn assert_dtype(dt: DType) -> AssertDType {
    match dt {
        DType::F32 => AssertDType::F32,
        DType::F16 => AssertDType::F16,
        DType::BF16 => AssertDType::BF16,
        DType::F64 => AssertDType::F64,
        DType::U8 => AssertDType::U8,
        DType::U32 => AssertDType::U32,
        DType::I64 => AssertDType::I64,
        DType::F8E4M3 => AssertDType::F8E4M3,
    }
}

/// Fold `t`'s statistics into `name`'s slot. Never synchronises, never
/// allocates, never fails loudly — a probe must not be able to break the
/// program it observes, so every internal error is logged and swallowed.
pub fn assert_tensor(t: &LiveTensor<'_>, name: &str) {
    if let Err(e) = try_assert_tensor(t, name) {
        tracing::warn!(
            target: "candle_core::tensor_assert",
            tensor = %name, error = %e,
            "tensor_assert: assert skipped"
        );
    }
}

fn try_assert_tensor(t: &LiveTensor<'_>, name: &str) -> Result<()> {
    let Some(slot_idx) = names::slot_for(name) else {
        return Ok(());
    };
    let elem_count = t.elem_count();
    if elem_count == 0 {
        return Ok(());
    }
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cuda) = &*storage else {
        // CPU and Metal tensors are out of scope: the whole point of this
        // instrument is to observe device work without synchronising it.
        return Ok(());
    };
    let dev = cuda.device.clone();
    let stream = dev.cuda_stream();

    // `is_contiguous` is the fast path: one linear walk. Otherwise the
    // kernel indexes through dims+strides, which is why nothing here has to
    // materialise a layout — a `contiguous()` at an assert site would be the
    // allocate-plus-copy this design exists to avoid.
    let dims = layout.dims();
    let (num_dims, dims_v, strides_v) = if layout.is_contiguous() {
        (0usize, Vec::new(), Vec::new())
    } else {
        if dims.len() > MAX_DIMS {
            crate::bail!(
                "rank {} exceeds the assert kernel's {MAX_DIMS}-dim strided path",
                dims.len()
            );
        }
        (
            dims.len(),
            dims.iter().map(|&d| d as i64).collect::<Vec<_>>(),
            layout.stride().iter().map(|&s| s as i64).collect::<Vec<_>>(),
        )
    };

    let elem_sz = t.dtype().size_in_bytes();
    let base = cuda.slice.device_ptr(&stream);
    let src = base + (layout.start_offset() * elem_sz) as u64;

    slots::with_slots(&dev, |sl| {
        let slot = sl.slot_ptr(slot_idx, &stream);
        let seq = sl.seq_ptr(&stream);
        unsafe {
            crate::set_kernel_breadcrumb("run_tensor_assert", file!(), line!());
            run_tensor_assert(
                src as *const std::ffi::c_void,
                assert_dtype(t.dtype()) as i32,
                elem_count as i64,
                num_dims as i32,
                if dims_v.is_empty() {
                    std::ptr::null()
                } else {
                    dims_v.as_ptr()
                },
                if strides_v.is_empty() {
                    std::ptr::null()
                } else {
                    strides_v.as_ptr()
                },
                slot as *mut std::ffi::c_void,
                seq as *mut std::ffi::c_void,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    })
}

/// Fold a raw device buffer's statistics into `name`'s slot.
///
/// The escape hatch for the things that are not tensors: a kernel workspace —
/// tile tables, permutations, expert ids, bucket offsets — is a bare
/// `CudaSlice`, and those are precisely where an out-of-range INDEX hides. An
/// index fault never produces a NaN where it happens; it produces one much
/// later, after a GEMM has read the wrong weights, so the only way to catch it
/// at the source is to look at the indices' range.
///
/// # Safety
///
/// `ptr` must name at least `elem_count` elements of `dtype` on `device`, live
/// for the duration of the launch. The kernel only reads.
pub unsafe fn assert_device_ptr(
    name: &str,
    ptr: u64,
    dtype: AssertDType,
    elem_count: usize,
    device: &crate::cuda_backend::CudaDevice,
) {
    if let Err(e) = try_assert_device_ptr(name, ptr, dtype, elem_count, device) {
        tracing::warn!(
            target: "candle_core::tensor_assert",
            buffer = %name, error = %e,
            "tensor_assert: raw assert skipped"
        );
    }
}

fn try_assert_device_ptr(
    name: &str,
    ptr: u64,
    dtype: AssertDType,
    elem_count: usize,
    device: &crate::cuda_backend::CudaDevice,
) -> Result<()> {
    let Some(slot_idx) = names::slot_for(name) else {
        return Ok(());
    };
    if elem_count == 0 || ptr == 0 {
        return Ok(());
    }
    let stream = device.cuda_stream();
    slots::with_slots(device, |sl| {
        let slot = sl.slot_ptr(slot_idx, &stream);
        let seq = sl.seq_ptr(&stream);
        unsafe {
            crate::set_kernel_breadcrumb("run_tensor_assert", file!(), line!());
            run_tensor_assert(
                ptr as *const std::ffi::c_void,
                dtype as i32,
                elem_count as i64,
                0,
                std::ptr::null(),
                std::ptr::null(),
                slot as *mut std::ffi::c_void,
                seq as *mut std::ffi::c_void,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    })
}

/// Fold a raw quantized ACTIVATION buffer's dequantized statistics into
/// `name`'s slot.
///
/// The int8 operands the MoE's grouped GEMMs consume are not `QTensor`s — they
/// are bare device buffers in a wave arena — and they are the only quantized
/// values on that path that are not weights. Reading them matters because a
/// quantized activation carries per-group SCALES: the quants themselves are
/// integers and cannot be non-finite, so anything wrong with the operand is
/// wrong with a scale, and a bad scale is invisible until it has been
/// multiplied through a GEMM.
///
/// Uses the same staging scratch and the same default-stream dequant as
/// [`assert_qtensor`], so it carries the same placement rule.
///
/// # Safety
///
/// `ptr` must name a complete `qtype` buffer of `elem_count` logical elements
/// on `device`, live for the duration of the launch.
pub unsafe fn assert_device_quant(
    name: &str,
    ptr: u64,
    qtype: i32,
    elem_count: usize,
    device: &crate::cuda_backend::CudaDevice,
) {
    if let Err(e) = try_assert_device_quant(name, ptr, qtype, elem_count, device) {
        tracing::warn!(
            target: "candle_core::tensor_assert",
            buffer = %name, error = %e,
            "tensor_assert: quantized-activation assert skipped"
        );
    }
}

fn try_assert_device_quant(
    name: &str,
    ptr: u64,
    qtype: i32,
    elem_count: usize,
    device: &crate::cuda_backend::CudaDevice,
) -> Result<()> {
    let Some(slot_idx) = names::slot_for(name) else {
        return Ok(());
    };
    if elem_count == 0 || ptr == 0 {
        return Ok(());
    }
    let stream = device.cuda_stream();
    scratch::with_f32_scratch(device, elem_count, |scratch_ptr| {
        scratch::dequantize_flat_into(ptr, scratch_ptr, elem_count, qtype);
        slots::with_slots(device, |sl| {
            let slot = sl.slot_ptr(slot_idx, &stream);
            let seq = sl.seq_ptr(&stream);
            unsafe {
                crate::set_kernel_breadcrumb("run_tensor_assert", file!(), line!());
                run_tensor_assert(
                    scratch_ptr as *const std::ffi::c_void,
                    AssertDType::F32 as i32,
                    elem_count as i64,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    slot as *mut std::ffi::c_void,
                    seq as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
            Ok(())
        })
    })
}

/// Fold a quantized tensor's DEQUANTIZED statistics into `name`'s slot.
///
/// The values a quantized weight actually contributes to a matmul are its
/// dequantized ones, so that is what this measures — a NaN scale and a
/// finite-but-enormous weight are different faults and only the dequantized
/// view distinguishes them. Dequant stages through a per-device grow-to-fit
/// scratch (see [`scratch`]) rather than allocating per call.
///
/// Runs the dequant kernels on the default stream, so this belongs at load time
/// or an epoch boundary, not inside a wave. Prefer
/// [`LiveQTensor::assert_once`].
pub fn assert_qtensor(q: &LiveQTensor<'_>, name: &str) {
    if let Err(e) = try_assert_qtensor(q, name) {
        tracing::warn!(
            target: "candle_core::tensor_assert",
            tensor = %name, error = %e,
            "tensor_assert: quantized assert skipped"
        );
    }
}

fn try_assert_qtensor(q: &LiveQTensor<'_>, name: &str) -> Result<()> {
    let Some(slot_idx) = names::slot_for(name) else {
        return Ok(());
    };
    let elem_count = q.shape().elem_count();
    if elem_count == 0 {
        return Ok(());
    }
    let QStorage::Cuda(cuda) = q.storage() else {
        return Ok(());
    };
    let dtype = cuda.dtype();
    let dev = cuda.device().clone();
    let stream = dev.cuda_stream();

    // MXFP4_KO is a GPU-only lane-major weight with no dequant kernel in either
    // family — `run_dequantize_block` does not know the layout and
    // `run_dequantize_ko` has no arm for it (the repack is an exact host-side
    // byte permutation, never a requant). Saying so is the point: a probe that
    // silently reported such a tensor clean would be worse than no probe.
    if dtype == GgmlDType::MXFP4_KO {
        crate::bail!("{dtype:?} has no dequant path, so its values cannot be examined");
    }

    scratch::with_f32_scratch(&dev, elem_count, |scratch_ptr| {
        let src = cuda.data_ptr();
        scratch::dequantize_into(src, scratch_ptr, q.shape(), dtype)?;
        slots::with_slots(&dev, |sl| {
            let slot = sl.slot_ptr(slot_idx, &stream);
            let seq = sl.seq_ptr(&stream);
            unsafe {
                crate::set_kernel_breadcrumb("run_tensor_assert", file!(), line!());
                run_tensor_assert(
                    scratch_ptr as *const std::ffi::c_void,
                    AssertDType::F32 as i32,
                    elem_count as i64,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    slot as *mut std::ffi::c_void,
                    seq as *mut std::ffi::c_void,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
            Ok(())
        })
    })
}

/// Examine `t` **now** and run `on_bad` if it holds a NaN or an Inf.
///
/// The synchronous counterpart to [`assert_tensor`], and the difference is the
/// whole reason it exists. `assert_tensor` folds into a slot that is not read
/// until the wave drains, which is correct for finding *where* a fault appears
/// and useless for capturing *what caused it*: by the time a drain speaks, the
/// operands are gone with the arena, and the next pass through the same site
/// carries different ones. A capture that has to hold the inputs and the output
/// together has to happen between the two, in the same pass, which means paying
/// for the answer immediately.
///
/// It costs one launch and one device synchronisation per call, so it is for a
/// site already known to fail — not for scattering across a forward. Placing it
/// on every layer reintroduces exactly the serialisation that hides
/// ordering-dependent faults.
///
/// `on_bad` receives the finding for this tensor alone. Returns whether it was
/// bad, so a caller that would rather branch than close over state can.
pub fn check_now(t: &LiveTensor<'_>, name: &str, on_bad: impl FnOnce(&Finding)) -> bool {
    match try_check_now(t, name, on_bad) {
        Ok(bad) => bad,
        Err(e) => {
            tracing::warn!(
                target: "candle_core::tensor_assert",
                tensor = %name, error = %e,
                "tensor_assert: synchronous check skipped"
            );
            false
        }
    }
}

fn try_check_now(
    t: &LiveTensor<'_>,
    name: &str,
    on_bad: impl FnOnce(&Finding),
) -> Result<bool> {
    let Some(slot_idx) = names::slot_for(name) else {
        return Ok(false);
    };
    let Device::Cuda(dev) = t.device() else {
        return Ok(false);
    };
    // Fold into the slot, then read back only THIS slot's statistics. The whole
    // table is copied because it is 128 KiB and one transfer is cheaper than
    // teaching the drain to address a single entry.
    try_assert_tensor(t, name)?;
    let stream = dev.cuda_stream();
    stream
        .synchronize()
        .map_err(|e| crate::Error::Msg(format!("tensor_assert: check fence: {e}")))?;
    let raw = slots::with_slots(dev, |sl| sl.read(dev))?;
    let Some(slot) = raw.get(slot_idx) else {
        return Ok(false);
    };
    if !slot.is_bad() {
        return Ok(false);
    }
    let finding = Finding {
        name: name.to_string(),
        seq: (slot.seq != 0 && slot.seq != u32::MAX).then_some(slot.seq),
        nan: slot.nan,
        inf: slot.inf,
        min: slot.min(),
        max: slot.max(),
        elems: slot.elems,
    };
    on_bad(&finding);
    Ok(true)
}

/// [`check_now`] for a raw quantized ACTIVATION buffer.
///
/// The int8 operands the MoE consumes are not tensors, and they are exactly
/// where a checkpoint chain runs out of things it can look at: the residual is
/// finite, the weights are finite, and between them sits a buffer that only
/// this can read. The quants are integers and cannot be non-finite, so anything
/// wrong here is wrong with a per-group scale.
///
/// # Safety
///
/// `ptr` must name a complete `qtype` buffer of `elem_count` logical elements
/// on `device`.
pub unsafe fn check_now_quant(
    name: &str,
    ptr: u64,
    qtype: i32,
    elem_count: usize,
    device: &crate::cuda_backend::CudaDevice,
    on_bad: impl FnOnce(&Finding),
) -> bool {
    match try_check_now_quant(name, ptr, qtype, elem_count, device, on_bad) {
        Ok(bad) => bad,
        Err(e) => {
            tracing::warn!(
                target: "candle_core::tensor_assert",
                buffer = %name, error = %e,
                "tensor_assert: synchronous quantized check skipped"
            );
            false
        }
    }
}

fn try_check_now_quant(
    name: &str,
    ptr: u64,
    qtype: i32,
    elem_count: usize,
    device: &crate::cuda_backend::CudaDevice,
    on_bad: impl FnOnce(&Finding),
) -> Result<bool> {
    let Some(slot_idx) = names::slot_for(name) else {
        return Ok(false);
    };
    unsafe { assert_device_quant(name, ptr, qtype, elem_count, device) };
    let stream = device.cuda_stream();
    stream
        .synchronize()
        .map_err(|e| crate::Error::Msg(format!("tensor_assert: quant check fence: {e}")))?;
    let raw = slots::with_slots(device, |sl| sl.read(device))?;
    let Some(slot) = raw.get(slot_idx).filter(|s| s.is_bad()) else {
        return Ok(false);
    };
    on_bad(&Finding {
        name: name.to_string(),
        seq: (slot.seq != 0 && slot.seq != u32::MAX).then_some(slot.seq),
        nan: slot.nan,
        inf: slot.inf,
        min: slot.min(),
        max: slot.max(),
        elems: slot.elems,
    });
    Ok(true)
}

/// Whether `name`'s one-shot latch is still unfired, claiming it if so.
pub fn should_run_once(name: &str) -> bool {
    match names::slot_for(name) {
        Some(idx) => names::claim_once(idx),
        None => false,
    }
}

/// Start a new epoch on `device`: clear every slot and re-arm every one-shot
/// latch, so the next wave's report describes that wave alone.
///
/// One small launch, no synchronisation.
pub fn epoch(device: &Device) -> Result<()> {
    let Device::Cuda(dev) = device else {
        return Ok(());
    };
    let stream = dev.cuda_stream();
    names::rearm_once();
    slots::with_slots(dev, |sl| sl.reset(&stream))
}

/// The device identity a report belongs to, for callers that hold a `Device`.
pub fn device_id(device: &Device) -> Option<DeviceId> {
    match device {
        Device::Cuda(d) => Some(d.id()),
        _ => None,
    }
}
