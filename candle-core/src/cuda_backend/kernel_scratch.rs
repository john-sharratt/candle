//! The allocator the CUDA launchers' own scratch is taken from.
//!
//! Some launchers keep a scratch they size from the launch shape — the INT8
//! decode, glue and prefill attention's split-KV partials, the provenance
//! scans' accumulators — and take it inside the launcher, where no host code
//! sees it (`candle-kernels`' `grow_scratch.cuh`). The launchers call back here
//! for it.
//!
//! # Inside a forward: a carve of the forward's own tier
//!
//! A forward opens a **scratch scope** ([`open_scratch_scope`]) when its
//! forward-wide span goes live, and closes it when that span rewinds. While it
//! is open, a launcher's block is carved from the span by the carver the span's
//! owner installs ([`install_scope_carve`]) — bytes the forward priced for
//! (`docs/held_ground_elimination.md` §3), gone with the forward. The scope's
//! open and close move the **scratch epoch**, so a launcher holding a block from
//! a forward that has ended takes a fresh one at its next launch rather than
//! reading a rewound span.
//!
//! What a scope asked for is remembered over the last [`RECENT_SCOPES`] forwards,
//! and [`scratch_reserve`] is what the next forward prices its span for. A carve
//! the span cannot hold falls through to held ground, so a forward that grew past
//! the reserve runs, and the next one is priced for it.
//!
//! # Outside a forward: held ground
//!
//! A [`Held`] block — the reservation's held ground, or a recorded pool
//! allocation when it has no room — that lives until its launcher replaces it or
//! the epoch moves, and is then retired through the device
//! ([`CudaDevice::retire`]): a launch already queued, or recorded into the segment
//! being captured, may still read it, so its range goes back only once that
//! segment has been issued.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use candle_kernels::paged_decode::{candle_set_kernel_scratch, candle_set_kernel_scratch_epoch};
use cudarc::driver::DevicePtr;

use super::held::Held;
use super::CudaDevice;

/// Carve `bytes` from the scratch scope open on device `ordinal`, or `None` when
/// its span cannot hold them.
pub type ScopeCarve = fn(ordinal: usize, bytes: usize) -> Option<u64>;

/// Forward scopes whose asks [`scratch_reserve`] remembers — the window the
/// span's growth negotiation keeps for the tier, for the same reason: a
/// prefill's chunks each ask a little more than the last.
const RECENT_SCOPES: usize = 16;

/// The boundary a reserve is rounded up to: the wave spans' own.
const RESERVE_ALIGN: usize = 256;

/// One device's scratch scope.
#[derive(Default)]
struct Scope {
    open: bool,
    /// Bytes the open scope's launchers asked for, carved or not.
    asked: usize,
    /// What each recent scope asked for, newest at `at`.
    recent: [usize; RECENT_SCOPES],
    at: usize,
}

/// Each device's handle, the held blocks its launchers hold by address, its
/// scope, and the installed carver.
#[derive(Default)]
struct Scratch {
    devices: HashMap<usize, CudaDevice>,
    blocks: HashMap<u64, Held<u8>>,
    scopes: HashMap<usize, Scope>,
    carve: Option<ScopeCarve>,
}

fn scratch() -> &'static Mutex<Scratch> {
    static SCRATCH: OnceLock<Mutex<Scratch>> = OnceLock::new();
    SCRATCH.get_or_init(|| Mutex::new(Scratch::default()))
}

fn lock() -> std::sync::MutexGuard<'static, Scratch> {
    scratch().lock().unwrap_or_else(|e| e.into_inner())
}

/// The epoch the launchers' blocks are valid for.
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// Move the epoch: every launcher takes its block afresh at its next launch.
fn bump_epoch() {
    let epoch = EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
    // SAFETY: a plain store into the kernels' global, read by the launchers on
    // the host before they take scratch.
    unsafe { candle_set_kernel_scratch_epoch(epoch) };
}

/// Serve `bytes` for a launcher's scratch on device `device`.
unsafe extern "C" fn take_kernel_scratch(device: i32, bytes: usize) -> *mut c_void {
    let ordinal = device as usize;
    // Decided under the lock, carved outside it: the carver takes the span's own
    // lock, and a scope opens under the span owner's lock and then takes this one.
    let carve = {
        let mut s = lock();
        let carve = s.carve;
        match s.scopes.get_mut(&ordinal).filter(|sc| sc.open) {
            Some(scope) => {
                scope.asked += bytes;
                carve
            }
            None => None,
        }
    };
    if let Some(ptr) = carve.and_then(|c| c(ordinal, bytes)) {
        return ptr as *mut c_void;
    }
    let Some(dev) = lock().devices.get(&ordinal).cloned() else {
        return std::ptr::null_mut();
    };
    // Growth may come from the pool, which this thread's recording segment
    // would take in: the segment is launched first, and the launch that asked
    // for the scratch lands behind it on the same stream.
    let Ok(_eager) = dev.pause_capture() else {
        return std::ptr::null_mut();
    };
    let stream = dev.cuda_stream();
    match Held::<u8>::alloc_on(&stream, bytes) {
        Ok(held) => {
            let ptr = held.device_ptr(&stream).0;
            lock().blocks.insert(ptr, held);
            ptr as *mut c_void
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Take back a block a launcher has replaced, and free it behind every launch
/// that may still read it. A carve is not here to free: its span rewinds with
/// the forward that owned it.
unsafe extern "C" fn release_kernel_scratch(device: i32, block: *mut c_void) {
    let (dev, held) = {
        let mut s = lock();
        let held = s.blocks.remove(&(block as u64));
        (s.devices.get(&(device as usize)).cloned(), held)
    };
    // Retired outside the lock: dropping a block gives its range back to the
    // ground, which takes the ground's own lock.
    if let (Some(dev), Some(held)) = (dev, held) {
        dev.retire(held);
    }
}

/// Register `dev` as its ordinal's handle and install the allocator. The first
/// handle registered for an ordinal is kept.
pub(crate) fn install(dev: &CudaDevice) {
    lock()
        .devices
        .entry(dev.cuda_stream().context().ordinal())
        .or_insert_with(|| dev.clone());
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        // SAFETY: plain function pointers with the signatures the launchers call.
        unsafe { candle_set_kernel_scratch(take_kernel_scratch, release_kernel_scratch) }
    });
}

/// Install the carver a scratch scope's blocks come from. Idempotent.
pub fn install_scope_carve(carve: ScopeCarve) {
    lock().carve = Some(carve);
}

/// A forward on device `ordinal` has opened its span: its launchers carve from it
/// until [`close_scratch_scope`].
pub fn open_scratch_scope(ordinal: usize) {
    {
        let mut s = lock();
        let scope = s.scopes.entry(ordinal).or_default();
        scope.open = true;
        scope.asked = 0;
    }
    bump_epoch();
}

/// The forward on device `ordinal` has let its span go: no launch may read a
/// block carved from it, and what it asked for is remembered for the next one.
pub fn close_scratch_scope(ordinal: usize) {
    {
        let mut s = lock();
        let scope = s.scopes.entry(ordinal).or_default();
        if scope.open {
            let at = scope.at;
            scope.recent[at] = scope.asked;
            scope.at = (at + 1) % RECENT_SCOPES;
            scope.open = false;
        }
    }
    bump_epoch();
}

/// Bytes a forward on device `ordinal` should price its span for, for its
/// launchers' scratch: the most any recent forward asked for, plus the eighth
/// `grow_scratch` asks over a shape.
pub fn scratch_reserve(ordinal: usize) -> usize {
    let s = lock();
    let widest = s
        .scopes
        .get(&ordinal)
        .map_or(0, |sc| sc.recent.iter().copied().max().unwrap_or(0));
    reserve_for(widest)
}

/// The reserve for a widest recent ask of `widest` bytes.
fn reserve_for(widest: usize) -> usize {
    (widest + widest / 8).next_multiple_of(RESERVE_ALIGN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;

    /// A block a launcher replaces is given back, and only that one: the
    /// launcher's current block stays held.
    #[test]
    fn a_replaced_block_is_released_and_the_current_one_kept() {
        let dev = CudaDevice::new(0).unwrap();
        let ordinal = dev.cuda_stream().context().ordinal() as i32;
        let held = |p: *mut c_void| lock().blocks.contains_key(&(p as u64));
        // SAFETY: the allocator's own entry points, as a launcher calls them.
        unsafe {
            let first = take_kernel_scratch(ordinal, 1 << 20);
            assert!(!first.is_null());
            let second = take_kernel_scratch(ordinal, 2 << 20);
            assert!(!second.is_null());
            assert!(held(first) && held(second));
            release_kernel_scratch(ordinal, first);
            assert!(!held(first), "the replaced block went back");
            assert!(held(second), "the current block is still the launcher's");
            release_kernel_scratch(ordinal, second);
        }
        dev.synchronize().unwrap();
    }

    /// The reserve is the widest recent ask and an eighth, on the span boundary.
    #[test]
    fn the_reserve_is_the_widest_recent_ask_and_an_eighth() {
        assert_eq!(reserve_for(0), 0);
        assert_eq!(reserve_for(8 << 20), 9 << 20);
        assert_eq!(reserve_for(1000), 1280, "1,125 rounds up to the boundary");
    }

    /// **A scope's carves come from its carver and are not held**, and what the
    /// scope asked for — carved or not — is what the next forward is priced for.
    /// Each open and close moves the epoch.
    ///
    /// On an ordinal no device has, so the fake carver can never serve a real
    /// launch from a test running beside this one; with no device, a carve the
    /// span refuses has nowhere else to go.
    #[test]
    fn a_scope_carves_and_remembers_what_it_asked_for() {
        const NO_DEVICE: usize = 15;
        fn carve(ordinal: usize, bytes: usize) -> Option<u64> {
            // A fixed address for one size, nothing for another: a span that
            // holds the first ask and not the second.
            (ordinal == NO_DEVICE && bytes == 4096).then_some(0x1000)
        }
        let previous = lock().carve;
        install_scope_carve(carve);
        let e0 = EPOCH.load(Ordering::Acquire);
        open_scratch_scope(NO_DEVICE);
        assert!(
            EPOCH.load(Ordering::Acquire) > e0,
            "opening moves the epoch"
        );
        // SAFETY: the allocator's own entry points, as a launcher calls them.
        unsafe {
            let carved = take_kernel_scratch(NO_DEVICE as i32, 4096);
            assert_eq!(carved as u64, 0x1000, "carved from the scope");
            assert!(!lock().blocks.contains_key(&0x1000), "a carve is not held");
            release_kernel_scratch(NO_DEVICE as i32, carved);
            let fell = take_kernel_scratch(NO_DEVICE as i32, 8192);
            assert!(
                fell.is_null(),
                "refused by the span, and no device to hold it"
            );
        }
        let e1 = EPOCH.load(Ordering::Acquire);
        close_scratch_scope(NO_DEVICE);
        assert!(EPOCH.load(Ordering::Acquire) > e1, "closing moves it again");
        assert_eq!(scratch_reserve(NO_DEVICE), reserve_for(4096 + 8192));
        lock().carve = previous;
    }
}
