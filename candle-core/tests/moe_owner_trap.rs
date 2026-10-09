//! **A VRAM entry naming a slot another expert holds traps in bucketize.**
//!
//! The owner check compares every VRAM entry bucketize snapshots for a GEMM
//! against the tag of the expert last installed in that slot. A mismatch is a
//! tile about to read one expert's weights under another's name, which would
//! otherwise surface as a wrong number layers later; the check ends the launch
//! in a trap instead, and the next synchronising call reports it.
//!
//! In a binary of its own: a trap poisons the CUDA context for the whole
//! process, which would fail every test that shared it.
#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use candle_core::quantized::cuda::{
    moe_bucketize, BucketizeLive, MoeBucketizeWorkspace, OwnerCheck,
};
use candle_core::quantized::decode_rows::DecodeRows;
use candle_core::{Device, Result, Tensor};

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
fn a_vram_entry_on_another_experts_slot_traps() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(dev) = &device else {
        unreachable!()
    };
    let n_experts = 8usize;
    // One row, every expert in VRAM: gate entries at `0x10_0000 + e · 0x100`,
    // slot `7 - e` of a zone of `0x100`-byte slots ending at `0x10_0800`.
    let mut table: Vec<u64> = (0..n_experts as u64)
        .map(|e| 0x10_0000 + e * 0x100)
        .collect();
    let gate = table.clone();
    table.extend(gate.iter().map(|g| g + 0x10));
    table.extend(gate.iter().map(|g| g + 0x20));
    let table_d = dev.memcpy_stod(&table)?;
    // Every slot tagged with its expert — except expert 2's slot, whose tag says
    // expert 6 of row 0 lives there.
    let (owners_h, owners_d) = mapped(n_experts * 4);
    for e in 0..n_experts {
        let tag = (1u32 << 16) | if e == 2 { 6 } else { e as u32 };
        unsafe { std::ptr::write_volatile((owners_h as *mut u32).add(n_experts - 1 - e), tag) };
    }
    let snap = dev.alloc_zeros::<u64>(3 * n_experts)?;
    let t = Tensor::from_vec(vec![2u32, 4], (1, 2), &device)?;
    let mut ws = MoeBucketizeWorkspace::new(dev, 1, 2)?;
    let stream = dev.cuda_stream();
    let live = BucketizeLive {
        gate_row: table_d.device_ptr(&stream).0,
        table_plane: n_experts as i64,
        snap: snap.device_ptr(&stream).0,
        pinned: [(0, 0), (0, 0)],
        summary: 0,
        summary_seq: 0,
        remote: 0,
        counters: 0,
        row: 0,
        promo: None,
        remote_dst: 0,
        started_rows: 0,
        ticket: 0,
        owner: OwnerCheck {
            owners: owners_d,
            zone_end: 0x10_0800,
            slot_bytes: 0x100,
            slots: n_experts as u32,
        },
    };
    moe_bucketize(
        &t,
        n_experts,
        32,
        &mut ws,
        Some(&live),
        &DecodeRows::prefix(1),
    )?;
    assert!(
        stream.synchronize().is_err(),
        "expert 2's entry points at a slot tagged for expert 6: the launch must trap"
    );
    Ok(())
}
