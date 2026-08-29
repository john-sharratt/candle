//! What the OS will *actually* page-lock, against what it *says* is available.
//!
//! The expert cache sizes its warm tier from `available_physical_ram()`, a
//! conservative estimate. This probe measures the real ceiling by allocating
//! pinned host memory in chunks until the driver refuses, so a sizing rule can
//! be judged against ground truth rather than against another estimate.
//!
//! Ignored: it deliberately takes most of the machine's free RAM for a few
//! seconds, so it must never run beside another GPU test.
//!
//! ```bash
//! cargo test -p candle-core --features cuda --test pinned_ceiling_probe -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use candle_core::{Device, Result};

const CHUNK: usize = 256 * 1024 * 1024;

struct Pinned(*mut std::ffi::c_void);

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost(self.0);
        }
    }
}

#[test]
#[ignore = "takes most of the machine's free RAM; run alone"]
fn how_much_will_the_os_actually_pin() -> Result<()> {
    // A device first: pinned allocation needs a context, and this also latches
    // the launch reading the engine sizes against.
    let _dev = Device::new_cuda(0)?;

    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let total = candle_core::vram::total_physical_ram().expect("total RAM probe");
    let avail = candle_core::vram::available_physical_ram().expect("available RAM probe");
    let launch = candle_core::vram::launch_available_ram().expect("launch RAM probe");
    // **The balloon only ever inflates inside the pinnable region.** Half the
    // machine, the same bound the warm tier obeys — the other half has to stay
    // pageable for the page cache, host shadows, and everything else alive on
    // the box. A probe that kept going past it would not find a ceiling, it
    // would find the point where the OS starts thrashing, and report that as
    // success.
    let cap = total / 2;

    println!("total            {:>8.2} GiB", gib(total));
    println!("available (live) {:>8.2} GiB", gib(avail));
    println!("available (launch){:>7.2} GiB", gib(launch));
    println!("half of RAM      {:>8.2} GiB   (the probe's cap)", gib(cap));
    println!(
        "chunk            {:>8.2} MiB",
        CHUNK as f64 / (1024.0 * 1024.0)
    );
    println!();

    let mut held: Vec<Pinned> = Vec::new();
    let mut taken: u64 = 0;
    let mut refused_at = None;
    while taken + CHUNK as u64 <= cap {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, CHUNK) };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            refused_at = Some(taken);
            break;
        }
        held.push(Pinned(ptr));
        taken += CHUNK as u64;
        if held.len().is_multiple_of(8) {
            let now = candle_core::vram::available_physical_ram().unwrap_or(0);
            println!(
                "  pinned {:>7.2} GiB   avail now {:>7.2} GiB",
                gib(taken),
                gib(now)
            );
        }
    }

    let after = candle_core::vram::available_physical_ram().unwrap_or(0);
    println!();
    println!("PINNED           {:>8.2} GiB", gib(taken));
    match refused_at {
        Some(at) => println!("driver refused at{:>8.2} GiB", gib(at)),
        None => println!("never refused — stopped at the half-RAM cap"),
    }
    println!("avail after      {:>8.2} GiB", gib(after));
    println!();
    println!(
        "VERDICT: pinned {:.2} GiB against a live estimate of {:.2} GiB — the estimate was \
         {:.0}% of what the OS gave.",
        gib(taken),
        gib(avail),
        100.0 * avail as f64 / taken.max(1) as f64,
    );

    drop(held);
    Ok(())
}
