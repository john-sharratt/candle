//! **A live launch whose cold expert never arrives gives it up and says so.**
//!
//! The live-table grouped GEMM's worker blocks wait for a cold expert's entry.
//! When the host cannot deliver it — it raises the abort word (a failed pack
//! read, a dead thread), or the wait passes the launch's spin limit (a drive
//! that stopped answering) — the waiting worker claims the mapped fault word
//! with the row, the expert and why, and gives the item up; any other cold wait
//! that sees the word set gives its expert up at once. The launch ends without
//! a trap, so the context stays usable: the next
//! synchronise succeeds and the host fails the forward from the fault word.
#![cfg(feature = "cuda")]

use std::ptr;
use std::sync::atomic::{fence, Ordering};
use std::thread;
use std::time::Duration;

use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use candle_core::quantized::cuda::{
    grouped_qmatmul_dev_q8a128, moe_bucketize, to_dynamic, BucketizeLive, DynamicActs,
    MoeBucketizeWorkspace, MoeLive, OwnerCheck,
};
use candle_core::quantized::decode_rows::DecodeRows;
use candle_core::quantized::{GgmlDType, Int8Mode, SumScale};
use candle_core::{DType, Device, Result, Tensor};
use candle_kernels::quantized::{MOE_FAULT_ABORTED, MOE_FAULT_EXPERT_SHIFT, MOE_FAULT_ROW_SHIFT};

/// `bytes` of mapped pinned host memory, zeroed: (host pointer, device address).
fn mapped(bytes: usize) -> (*mut std::ffi::c_void, u64) {
    let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut d: sys::CUdeviceptr = 0;
    unsafe {
        assert_eq!(
            sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP),
            sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            sys::cuMemHostGetDevicePointer_v2(&mut d, raw, 0),
            sys::CUresult::CUDA_SUCCESS
        );
        std::ptr::write_bytes(raw as *mut u8, 0, bytes);
    }
    (raw, d)
}

/// One live gate launch over a table where every expert is cold, on MoE row
/// `row`, with the given spin limit; `release` runs once the launch is queued.
/// Returns the fault word after the next synchronise, which must succeed.
fn cold_launch(row: i32, spin_limit_ns: u64, release: impl FnOnce(*mut u32)) -> Result<u64> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(dev) = &device else {
        unreachable!()
    };
    let (n_experts, nrows, ncols) = (8usize, 64usize, 1024usize);
    let (n_tokens, k) = (4usize, 2usize);
    let a_ub = n_tokens * k;

    // Every expert cold: every routed expert goes to the workers, which wait,
    // and nothing will publish them.
    let (_table_h, table_d) = mapped(3 * n_experts * 8);
    let (words_h, words_d) = mapped(16);
    let snap = dev.alloc_zeros::<u64>(3 * n_experts)?;
    let ids: Vec<u32> = (0..a_ub as u32).map(|i| i % n_experts as u32).collect();
    let t = Tensor::from_vec(ids, (n_tokens, k), &device)?;
    let x = Tensor::ones((a_ub, ncols), DType::F32, &device)?;
    let DynamicActs::Int8(op) = to_dynamic(&x, Int8Mode::Performance, dev, SumScale::Raw)? else {
        panic!("an int8 mode quantizes")
    };

    let stream = dev.cuda_stream();
    let summary = dev.alloc_zeros::<u32>(n_experts + 1)?;
    let remote = dev.alloc_zeros::<i32>(n_experts * 4)?;
    let counters = dev.alloc_zeros::<i32>(3)?;
    let scratch = dev.alloc_zeros::<u8>(8 * 65536)?;
    let mut ws = MoeBucketizeWorkspace::new(dev, n_tokens, k)?;
    let (sp, _g1) = summary.device_ptr(&stream);
    let (rp, _g2) = remote.device_ptr(&stream);
    let (cp, _g3) = counters.device_ptr(&stream);
    let (scp, _g4) = scratch.device_ptr(&stream);
    let (hp, _g5) = ws.header.device_ptr(&stream);
    let (np, _g6) = snap.device_ptr(&stream);
    let blive = BucketizeLive {
        gate_row: table_d,
        table_plane: n_experts as i64,
        snap: np,
        pinned: [(0, 0), (0, 0)],
        summary: sp,
        summary_seq: 1,
        remote: rp,
        counters: cp,
        row,
        promo: None,
        remote_dst: 0,
        started_rows: 0,
        ticket: 0,
        owner: OwnerCheck::default(),
        ahead: None,
    };
    drop((_g1, _g2, _g3, _g4, _g5, _g6));
    moe_bucketize(
        &t,
        n_experts,
        32,
        &mut ws,
        Some(&blive),
        &DecodeRows::prefix(n_tokens),
    )?;
    let live = MoeLive {
        abort: words_d,
        fault: words_d + 8,
        live_row: table_d,
        remote_dst: 0,
        dst_offset: 0,
        remote: rp,
        header: hp,
        counter: cp,
        scratch: scp,
        slot_bytes: 65536,
        stall: 0,
        ahead: 0,
        ahead_done: 0,
        spin_limit_ns,
        workers: 8,
        row,
    };
    let _out = grouped_qmatmul_dev_q8a128(
        &op,
        &snap,
        0,
        n_experts,
        GgmlDType::Q6_KO,
        nrows,
        &ws.tile_expert,
        &ws.tile_b_start,
        &ws.tile_b_cnt,
        a_ub,
        2,
        Some(&live),
        dev,
    )?;
    unsafe {
        let _ = sys::cuStreamQuery(stream.cu_stream());
    }
    release(words_h as *mut u32);
    stream.synchronize().map_err(candle_core::Error::wrap)?;
    // The context survived: work after the launch still runs.
    let after = (Tensor::ones(4, DType::F32, &device)? + 1.0)?.to_vec1::<f32>()?;
    assert_eq!(after, vec![2.0; 4], "the context is usable after a fault");
    Ok(unsafe { std::ptr::read_volatile((words_h as *const u8).add(8) as *const u64) })
}

/// The host's abort — a store to a mapped word, no driver call — ends the wait:
/// the fault word says aborted, on the launch's row, for a routed expert.
#[test]
fn an_aborted_cold_wait_claims_the_fault_word_and_leaves_the_context_usable() -> Result<()> {
    // Far above the test's own delay, so the abort — not the backstop — is
    // what ends the wait.
    let word = cold_launch(7, 60_000_000_000, |abort| {
        thread::sleep(Duration::from_millis(50));
        unsafe { ptr::write_volatile(abort, 1) };
        fence(Ordering::SeqCst);
    })?;
    assert_ne!(word & MOE_FAULT_ABORTED, 0, "aborted: {word:#x}");
    assert_eq!((word >> MOE_FAULT_ROW_SHIFT) & 0x7fff, 7, "row");
    assert!(
        (word >> MOE_FAULT_EXPERT_SHIFT) & 0xffff < 8,
        "a routed expert: {word:#x}"
    );
    Ok(())
}

/// A wait past the spin limit ends on its own: the fault word says not
/// aborted, with at least the limit waited.
#[test]
fn a_cold_wait_past_the_spin_limit_claims_the_fault_word() -> Result<()> {
    let word = cold_launch(2, 20_000_000, |_| {})?;
    assert_eq!(
        word & MOE_FAULT_ABORTED,
        0,
        "timed out, not aborted: {word:#x}"
    );
    assert_eq!((word >> MOE_FAULT_ROW_SHIFT) & 0x7fff, 2, "row");
    assert!(
        word & 0xffff_ffff >= 20_000,
        "waited at least the 20 ms limit: {word:#x}"
    );
    Ok(())
}
