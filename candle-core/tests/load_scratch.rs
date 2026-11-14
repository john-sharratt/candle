//! **A load's transient allocations land in its scratch and go back there.**
//!
//! The scratch is the top of the span, lent to a model load's transients until
//! the dense block is frozen (`cuda_backend::load_scratch`). This drives the one
//! path the unit tests cannot: cudarc's own allocator asking the installed watch
//! to serve, and its slice's drop handing the address back rather than freeing
//! it to the driver.
//!
//! In a binary of its own: the scratch is opened on the device's one stream,
//! and any other test's scoped allocation on that device would be served from it.
#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::result::{
    free_sync, malloc_sync, memcpy_dtoh_async,
};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::load_scratch::{self, ScratchReport};
use candle_core::{Device, Result};

#[test]
fn a_scoped_allocation_is_served_from_the_scratch_and_given_back() -> Result<()> {
    let Device::Cuda(dev) = Device::new_cuda(0)? else {
        unreachable!("new_cuda answers a CUDA device")
    };
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    const LEN: usize = 4 << 20;
    // The scratch's ground: a plain device allocation standing in for the
    // span's top.
    let base = unsafe { malloc_sync(LEN) }.expect("cuMemAlloc");
    load_scratch::open(ordinal, stream.cu_stream() as usize, base, LEN)?;

    let outside = dev.alloc_zeros::<u8>(1024)?;
    let (inside, again) = load_scratch::transient(|| -> Result<_> {
        Ok((dev.alloc_zeros::<u32>(1000)?, dev.alloc_zeros::<u8>(1)?))
    })?;
    let outside_ptr = outside.device_ptr(&stream).0;
    let inside_ptr = inside.device_ptr(&stream).0;
    let again_ptr = again.device_ptr(&stream).0;
    assert!(
        !(base..base + LEN as u64).contains(&outside_ptr),
        "an allocation outside the scope is the driver's"
    );
    assert_eq!(
        inside_ptr, base,
        "the first scoped take is the scratch's base"
    );
    assert_eq!(again_ptr, base + 4096, "the next one follows, 256-aligned");

    // The served memory is real device memory: zero-filled through it, read back.
    let mut host = vec![1u32; 1000];
    unsafe { memcpy_dtoh_async(&mut host, inside_ptr, stream.cu_stream()) }.expect("dtoh");
    stream.synchronize().expect("sync");
    assert!(host.iter().all(|&v| v == 0));

    // Close refuses while a served slice lives; dropping it hands it back.
    assert!(load_scratch::close(ordinal).is_err());
    drop(inside);
    drop(again);
    stream.synchronize().expect("sync");
    assert_eq!(
        load_scratch::close(ordinal)?,
        Some(ScratchReport {
            bytes: LEN,
            peak: 4096 + 256,
            spilled: 0
        })
    );
    drop(outside);
    unsafe { free_sync(base) }.expect("cuMemFree");
    Ok(())
}
