//! **A live launch whose cold expert can never be published ends in a sticky
//! error.**
//!
//! The live-table grouped GEMM's worker blocks wait for a cold expert's entry.
//! When the host cannot deliver it — a failed pack read, a dead thread — it
//! raises the abort word with a plain host store, and every waiting worker traps.
//! The next synchronising call must then report an error, so no result computed
//! from the layer is ever returned.
//!
//! In a binary of its own: a trap poisons the CUDA context for the whole
//! process, which would fail every test that shared it.
#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use candle_core::quantized::cuda::{
    grouped_qmatmul_dev_q8a128, moe_bucketize, to_dynamic, BucketizeLive, DynamicActs,
    MoeBucketizeWorkspace, MoeLive,
};
use candle_core::quantized::{GgmlDType, Int8Mode, SumScale};
use candle_core::{DType, Device, Result, Tensor};

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

#[test]
fn an_aborted_cold_wait_traps_and_the_next_sync_reports_it() -> Result<()> {
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
    let (abort_h, abort_d) = mapped(4);
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
        row: 0,
        promo: None,
        remote_dst: 0,
    };
    drop((_g1, _g2, _g3, _g4, _g5, _g6));
    moe_bucketize(&t, n_experts, 32, &mut ws, Some(&blive), n_tokens)?;
    let live = MoeLive {
        abort: abort_d,
        live_row: table_d,
        remote_dst: 0,
        dst_offset: 0,
        remote: rp,
        header: hp,
        counter: cp,
        scratch: scp,
        slot_bytes: 65536,
        stall: 0,
        // Far above the test's own delay, so the abort — not the backstop — is
        // what ends the wait.
        spin_limit_ns: 60_000_000_000,
        workers: 8,
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
    std::thread::sleep(std::time::Duration::from_millis(50));

    // The host's abort: a store to a mapped word — no driver call.
    unsafe { std::ptr::write_volatile(abort_h as *mut u32, 1) };
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let synced = stream.synchronize();
    assert!(
        synced.is_err(),
        "an aborted wait must surface as an error at the next synchronize"
    );
    Ok(())
}
