//! Prints the DXGI per-process VRAM budget for the CUDA device — the
//! OS-authoritative number the WDDM residency work is calibrated against.
//! Ignored by default: it needs a CUDA device and a Windows/WDDM host, and its
//! value is the printout, not an assertion.
#![cfg(all(feature = "cuda", windows))]

use candle_core::quantized::pinned_staging::PinnedBuf;
use candle_core::vram::{available_physical_ram, total_physical_ram, DxgiProbe, VramProbe};
use candle_core::Device;

const GIB: u64 = 1 << 30;

/// Pins host memory in 4 GiB steps to `total/2 − 1 GiB` — the expert warm
/// tier's page-lock ceiling — then uploads 14 MiB from PAGEABLE memory, which
/// needs the driver to lock a bounce buffer. Prints whether that upload lands at
/// each step.
#[test]
#[ignore]
fn print_pageable_upload_under_pinning() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let total = total_physical_ram().expect("ram probe");
    let ceiling = (total / 2).saturating_sub(GIB);
    let src = vec![7u8; 14 << 20];
    let stream = cuda.cuda_stream();
    let mut held = Vec::new();
    loop {
        let upload = stream
            .memcpy_stod(&src)
            .and_then(|d| stream.synchronize().map(|_| d));
        println!(
            "pinned {:>3} GiB: pageable upload {}",
            held.len() * 4,
            match &upload {
                Ok(_) => "ok".to_string(),
                Err(e) => format!("FAILED: {e:?}"),
            }
        );
        if (held.len() as u64 + 1) * 4 * GIB > ceiling {
            break;
        }
        held.push(PinnedBuf::alloc_owned_default(4 * GIB as usize)?);
    }
    Ok(())
}

/// Pins 80 GiB, then commits and touches pageable memory in 4 GiB steps while
/// available RAM stays above 6 GiB, printing the NON_LOCAL budget after each —
/// the measurement that says whether pageable growth shrinks the budget the
/// pinned part is already charged against.
#[test]
#[ignore]
fn print_non_local_budget_under_pageable_growth() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let probe = DxgiProbe::for_cuda_device(cuda)?;
    let report = |label: String| -> candle_core::Result<()> {
        let (budget, usage) = probe.non_local()?;
        let avail = available_physical_ram().unwrap_or(0);
        println!(
            "{label}: budget={:.2} GiB usage={:.2} GiB available={:.2} GiB",
            budget as f64 / GIB as f64,
            usage as f64 / GIB as f64,
            avail as f64 / GIB as f64
        );
        Ok(())
    };
    report("start".into())?;
    let pinned: Vec<PinnedBuf> = (0..20)
        .map(|_| PinnedBuf::alloc_owned_default(4 * GIB as usize))
        .collect::<candle_core::Result<_>>()?;
    report(format!("pinned {} GiB", pinned.len() * 4))?;
    let mut paged: Vec<Vec<u8>> = Vec::new();
    while available_physical_ram().unwrap_or(0) > 10 * GIB {
        paged.push(vec![1u8; 4 * GIB as usize]);
        report(format!("pageable {:>3} GiB", paged.len() * 4))?;
    }
    Ok(())
}

/// Pins host memory in 4 GiB steps until the driver refuses, printing the
/// NON_LOCAL segment's budget and usage after each — the measurement that says
/// whether page-locked memory is charged against that budget, and where the
/// driver stops.
#[test]
#[ignore]
fn print_non_local_budget_under_pinning() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let probe = DxgiProbe::for_cuda_device(cuda)?;
    let (budget, usage) = probe.non_local()?;
    println!(
        "non-local before: budget={:.2} GiB usage={:.2} GiB",
        budget as f64 / GIB as f64,
        usage as f64 / GIB as f64
    );
    let mut held = Vec::new();
    loop {
        match PinnedBuf::alloc_owned_default(4 * GIB as usize) {
            Ok(b) => held.push(b),
            Err(e) => {
                println!("refused at {} GiB pinned: {e}", held.len() * 4);
                break;
            }
        }
        let (budget, usage) = probe.non_local()?;
        println!(
            "pinned {:>3} GiB: budget={:.2} GiB usage={:.2} GiB",
            held.len() * 4,
            budget as f64 / GIB as f64,
            usage as f64 / GIB as f64
        );
    }
    Ok(())
}

/// Pins host memory in 2 GiB steps until the driver refuses, and after each
/// step uploads 64 MiB to the device FROM the newest pinned block — a direct
/// DMA, which needs the GPU to address that block — printing whether it lands
/// and the NON_LOCAL budget and usage. The measurement that says whether a
/// pinned block the driver granted can still be read by the GPU once the
/// process's page-locked total passes the segment's budget.
#[test]
#[ignore]
fn print_pinned_dma_under_pinning() -> candle_core::Result<()> {
    use cudarc::driver::sys;
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let probe = DxgiProbe::for_cuda_device(cuda)?;
    let stream = cuda.cuda_stream();
    const STEP: usize = 2 << 30;
    const COPY: usize = 64 << 20;
    let dst = unsafe { stream.alloc::<u8>(COPY) }.map_err(candle_core::Error::wrap)?;
    let dst_ptr = {
        use cudarc::driver::DevicePtr;
        dst.device_ptr(&stream).0
    };
    let mut held: Vec<*mut std::ffi::c_void> = Vec::new();
    loop {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a plain page-locked host allocation, freed below.
        let granted = unsafe { sys::cuMemAllocHost_v2(&mut ptr, STEP) };
        if granted != sys::CUresult::CUDA_SUCCESS {
            println!(
                "refused at {} GiB pinned: {granted:?}",
                held.len() * STEP / (1 << 30)
            );
            break;
        }
        held.push(ptr);
        // SAFETY: `ptr` holds STEP bytes; the device buffer holds COPY.
        let dma = unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst_ptr, ptr, COPY, stream.cu_stream())
                .result()
                .and_then(|_| sys::cuStreamSynchronize(stream.cu_stream()).result())
        };
        let (budget, usage) = probe.non_local()?;
        println!(
            "pinned {:>3} GiB: dma {} | non-local budget={:.2} GiB usage={:.2} GiB available RAM={:.2} GiB",
            held.len() * STEP / (1 << 30),
            match dma {
                Ok(()) => "ok".to_string(),
                Err(e) => format!("FAILED: {e:?}"),
            },
            budget as f64 / GIB as f64,
            usage as f64 / GIB as f64,
            available_physical_ram().unwrap_or(0) as f64 / GIB as f64,
        );
    }
    for p in held {
        // SAFETY: each came from `cuMemAllocHost_v2` above.
        unsafe { sys::cuMemFreeHost(p) };
    }
    Ok(())
}

#[test]
#[ignore]
fn print_dxgi_budget() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let probe = DxgiProbe::for_cuda_device(cuda)?;
    let r = probe.read()?;
    println!(
        "dxgi: total={:.2}GB ({}MiB) headroom(Budget-CurrentUsage)={:.2}GB ({}MiB)",
        r.total as f64 / 1e9,
        r.total / (1024 * 1024),
        r.headroom as f64 / 1e9,
        r.headroom / (1024 * 1024),
    );
    Ok(())
}
