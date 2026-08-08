//! `Backing::Lease` — tensors that view device memory owned elsewhere.
//!
//! An arena slot's bytes belong to a reservation held for the process lifetime,
//! so letting `CudaSlice::drop` reach `cuMemFreeAsync` on them would be a
//! correctness error rather than a leak. These tests pin the properties the KV
//! cache relies on: a lease reads the memory it was pointed at, a write through
//! a lease is visible to the owner, dropping a lease leaves the owner's memory
//! intact, and `to_owned_qtensor` — the only sanctioned way to outlive a lease
//! — really copies rather than handing back another view.
//!
//! The *lifetime* half of the contract needs no runtime test: `LiveQTensor<'w>`
//! is covariant in `'w`, and the `compile_fail` doctests on that type prove a
//! lease cannot widen to `'static` or reach [`candle_core::quantized::QMatMul`].
//!
//! See `docs/archived/arena_unification.md` §3.7.
#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};

/// Device pointer of a contiguous F32 tensor's storage.
fn f32_ptr(t: &Tensor) -> u64 {
    let (storage, _) = t.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(c) => {
            let slice = c.as_cuda_slice::<f32>().expect("f32 slice");
            let stream = c.device.cuda_stream();
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr
        }
        _ => panic!("expected CUDA storage"),
    }
}

#[test]
fn lease_reads_the_owners_bytes() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let owner = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), &dev)?;
    let leased = unsafe { Tensor::from_leased_cuda_ptr(f32_ptr(&owner), DType::F32, (4,), &dev)? };
    assert_eq!(leased.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

/// A lease is a view, not a copy: the owner sees writes made through it. This
/// is what `write_contiguous` depends on when it stores a band into a slot.
#[test]
fn writes_through_a_lease_reach_the_owner() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let owner = Tensor::zeros((4,), DType::F32, &dev)?;
    let leased = unsafe { Tensor::from_leased_cuda_ptr(f32_ptr(&owner), DType::F32, (4,), &dev)? };
    let src = Tensor::from_vec(vec![7.0f32, 8.0], (2,), &dev)?;
    leased.slice_set(&src, 0, 1)?;
    assert_eq!(owner.to_vec1::<f32>()?, vec![0.0, 7.0, 8.0, 0.0]);
    Ok(())
}

/// **The load-bearing one.** Dropping a lease must not free the owner's memory.
/// If `Drop for CudaStorage` were missing, or dispatched on the wrong backing,
/// the pool would reclaim these bytes and the subsequent read would be garbage
/// or a fault.
#[test]
fn dropping_a_lease_leaves_the_owner_intact() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let owner = Tensor::from_vec((0..64).map(|i| i as f32).collect::<Vec<_>>(), (64,), &dev)?;
    let ptr = f32_ptr(&owner);

    // Churn leases so a missing drop-suppression shows up as a use-after-free
    // rather than as a merely-suspicious single case. Each iteration also
    // allocates, so a wrongly-freed slot would likely be handed straight back.
    for _ in 0..64 {
        let leased = unsafe { Tensor::from_leased_cuda_ptr(ptr, DType::F32, (64,), &dev)? };
        assert_eq!(leased.to_vec1::<f32>()?[7], 7.0);
        drop(leased);
        let _scratch = Tensor::zeros((1024,), DType::F32, &dev)?;
    }

    let got = owner.to_vec1::<f32>()?;
    assert_eq!(got.len(), 64);
    for (i, v) in got.iter().enumerate() {
        assert_eq!(*v, i as f32, "owner byte {i} was corrupted by a lease drop");
    }
    Ok(())
}

/// A lease survives being reshaped and narrowed — views share the `Arc<Storage>`
/// so the lease travels with them, and none of the derived tensors free it.
#[test]
fn lease_travels_with_views() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let owner = Tensor::from_vec((0..12).map(|i| i as f32).collect::<Vec<_>>(), (12,), &dev)?;
    let ptr = f32_ptr(&owner);
    {
        let leased = unsafe { Tensor::from_leased_cuda_ptr(ptr, DType::F32, (3, 4), &dev)? };
        let row = leased.narrow(0, 1, 1)?;
        assert_eq!(
            row.flatten_all()?.to_vec1::<f32>()?,
            vec![4.0, 5.0, 6.0, 7.0]
        );
    }
    assert_eq!(owner.to_vec1::<f32>()?[11], 11.0);
    Ok(())
}

/// Leases are typed views over raw bytes: the same slot read as `U8` sees the
/// float's little-endian representation. This is what lets one byte-slab arena
/// serve every KV format.
#[test]
fn a_slot_can_be_viewed_as_bytes() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let owner = Tensor::from_vec(vec![1.0f32], (1,), &dev)?;
    let bytes = unsafe { Tensor::from_leased_cuda_ptr(f32_ptr(&owner), DType::U8, (4,), &dev)? };
    // 1.0f32 == 0x3F800000, little-endian.
    assert_eq!(bytes.to_vec1::<u8>()?, vec![0x00, 0x00, 0x80, 0x3F]);
    Ok(())
}

/// A quantized lease addresses the owner's blocks, not a copy of them.
///
/// `QCudaStorage::clone` is a device-to-device copy, so the ordinary way to
/// keep a quantized view around would silently duplicate the arena. The lease
/// is what makes `quantize_into` write where the caller meant.
#[test]
fn a_quantized_lease_addresses_the_owners_blocks() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &dev else {
        panic!("expected a CUDA device")
    };
    let src = Tensor::from_vec((0..64).map(|i| i as f32).collect::<Vec<_>>(), (64,), &dev)?;
    let owner = QTensor::quantize(&src, GgmlDType::Q8_0)?;
    let owner_ptr = owner.cuda_data_ptr().expect("owner is CUDA-resident");

    let leased = unsafe { QTensor::from_leased_cuda_ptr(owner_ptr, GgmlDType::Q8_0, 64, cuda)? };
    assert_eq!(
        leased.cuda_data_ptr(),
        Some(owner_ptr),
        "a lease must address the owner's bytes, not a copy"
    );
    assert_eq!(
        &*leased.data()?,
        &*owner.data()?,
        "and therefore read the owner's blocks verbatim"
    );
    Ok(())
}

/// `to_owned_qtensor` is the sanctioned way out of a lease's lifetime, and it
/// really does allocate — the whole point is that the copy survives the owner.
#[test]
fn to_owned_qtensor_copies_off_the_lease() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &dev else {
        panic!("expected a CUDA device")
    };
    let src = Tensor::from_vec((0..64).map(|i| i as f32).collect::<Vec<_>>(), (64,), &dev)?;
    let owner = QTensor::quantize(&src, GgmlDType::Q8_0)?;
    let owner_ptr = owner.cuda_data_ptr().expect("owner is CUDA-resident");

    let escaped = {
        let leased =
            unsafe { QTensor::from_leased_cuda_ptr(owner_ptr, GgmlDType::Q8_0, 64, cuda)? };
        leased.to_owned_qtensor()
    };
    assert_ne!(
        escaped.cuda_data_ptr(),
        Some(owner_ptr),
        "the copy must have its own allocation, or it is still a lease"
    );
    assert_eq!(
        &*escaped.data()?,
        &*owner.data()?,
        "with the owner's bytes reproduced exactly"
    );
    Ok(())
}
