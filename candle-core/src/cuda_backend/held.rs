//! Long-lived device scratch, held from the reservation rather than the pool.
//!
//! Some device buffers outlive any one forward and are reused across all of
//! them: a launch-table staging scratch, a sampler's argument pack, a resident
//! ring of slot headers, an attention kernel's split partials. Each grows to the
//! widest request it sees and is then kept. Taken from the CUDA pool, every one
//! of them was memory outside the reservation — invisible to the partition the
//! KV side and the weight side negotiate over, and the first thing WDDM demotes
//! when the card fills.
//!
//! [`Held`] is such a buffer taken from **held ground**: a block of the span
//! the reservation keeps for exactly this, installed by the crate that owns the
//! span ([`install_held_ground`]). This crate cannot name the span, so the
//! ground is a trait object, the same inversion `wave_provenance` uses for the
//! wave arenas.
//!
//! # Release is stream-ordered
//!
//! Dropping a `Held` gives its range back to the ground, which must not hand it
//! to anyone else while a launch already queued may still read it. The ground
//! fences the compute stream at release and reuses the range once the fence has
//! fired. A holder whose buffer a *recorded* launch may still read drops it
//! through [`super::CudaDevice::retire`], so the release happens after that
//! segment has been issued — the same rule every device free here follows.
//!
//! # When there is no ground
//!
//! Before the reservation exists, or when the ground is full, the buffer is an
//! ordinary pool allocation and [`crate::forbidden_alloc`] records it — never a
//! failure, because the buffers here are what keep a forward running, and a
//! report that names the shortfall is the useful answer.

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::panic::Location;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};

use super::stream_alloc::stream_alloc;
use super::{CudaDevice, WrapErr};
use crate::Result;

/// Device memory a reservation keeps for held buffers.
pub trait HeldGround: Send + Sync {
    /// `bytes` at an address aligned to 256 for the holder at `site`, or `None`
    /// when the ground cannot hold them now. `site` is what the ground's
    /// ledger files the range — or the refusal — under.
    fn take(&self, bytes: usize, site: &'static Location<'static>) -> Option<u64>;
    /// Give back the range [`Self::take`] returned at `ptr`. Launches queued on
    /// `stream` before this call may still read it; the ground reuses it only
    /// once they have run.
    fn give_back(&self, ptr: u64, stream: &Arc<CudaStream>);
}

fn grounds() -> &'static Mutex<HashMap<usize, Arc<dyn HeldGround>>> {
    static GROUNDS: OnceLock<Mutex<HashMap<usize, Arc<dyn HeldGround>>>> = OnceLock::new();
    GROUNDS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install the held ground for device `ordinal`. Replaces any earlier one: a
/// buffer taken from the old ground keeps the handle it was taken with and
/// gives its range back there.
pub fn install_held_ground(ordinal: usize, ground: Arc<dyn HeldGround>) {
    grounds().lock().unwrap().insert(ordinal, ground);
}

fn ground_for(ordinal: usize) -> Option<Arc<dyn HeldGround>> {
    grounds().lock().unwrap().get(&ordinal).cloned()
}

/// Where a [`Held`]'s memory came from.
enum Origin {
    /// A range of held ground, given back on drop.
    Ground {
        ground: Arc<dyn HeldGround>,
        stream: Arc<CudaStream>,
    },
    /// A pool allocation, freed on drop like any slice.
    Pool,
}

/// A long-lived device buffer of `len` elements of `T`, from held ground when
/// the device has some to give and from the pool otherwise.
///
/// Reads as a [`CudaSlice<T>`]. Its contents are uninitialised: the holder
/// writes every element it reads.
pub struct Held<T: DeviceRepr> {
    slice: ManuallyDrop<CudaSlice<T>>,
    origin: Origin,
}

impl<T: DeviceRepr> Held<T> {
    /// `len` elements for `dev`, released on its compute stream.
    ///
    /// # Safety
    ///
    /// The memory is uninitialised — another holder's bytes when it comes from
    /// ground — and must be written before it is read.
    ///
    /// The ground files the range under the caller's location, so a buffer is
    /// named by the code that asked for it; a helper that allocates on its
    /// callers' behalf carries `#[track_caller]` too.
    #[track_caller]
    pub unsafe fn alloc(dev: &CudaDevice, len: usize) -> Result<Self> {
        Self::alloc_on(&dev.compute_stream(), len)
    }

    /// `len` elements whose last reader runs on `stream` — the stream the
    /// release is fenced on.
    ///
    /// # Safety
    ///
    /// As [`Self::alloc`].
    #[track_caller]
    pub unsafe fn alloc_on(stream: &Arc<CudaStream>, len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let ordinal = stream.context().ordinal();
        if let Some(ground) = ground_for(ordinal) {
            if let Some(ptr) = ground.take(bytes.max(1), Location::caller()) {
                let slice = stream.upgrade_device_ptr::<T>(ptr, len);
                return Ok(Self {
                    slice: ManuallyDrop::new(slice),
                    origin: Origin::Ground {
                        ground,
                        stream: stream.clone(),
                    },
                });
            }
        }
        Ok(Self {
            slice: ManuallyDrop::new(stream_alloc::<T>(stream, len)?),
            origin: Origin::Pool,
        })
    }

    /// A held copy of `data`, uploaded on `dev`'s compute stream.
    #[track_caller]
    pub fn from_host(dev: &CudaDevice, data: &[T]) -> Result<Self> {
        // SAFETY: every element is written by the upload below.
        let mut held = unsafe { Self::alloc(dev, data.len())? };
        if !data.is_empty() {
            let _eager = dev.pause_capture()?;
            dev.compute_stream().memcpy_htod(data, &mut *held).w()?;
        }
        Ok(held)
    }

    /// Whether the buffer stands on held ground rather than the pool.
    pub fn on_ground(&self) -> bool {
        matches!(self.origin, Origin::Ground { .. })
    }
}

impl<T: DeviceRepr> Deref for Held<T> {
    type Target = CudaSlice<T>;
    fn deref(&self) -> &CudaSlice<T> {
        &self.slice
    }
}

impl<T: DeviceRepr> DerefMut for Held<T> {
    fn deref_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

impl<T: DeviceRepr> Drop for Held<T> {
    fn drop(&mut self) {
        // SAFETY: taken exactly once, here, and not touched again.
        let slice = unsafe { ManuallyDrop::take(&mut self.slice) };
        match &self.origin {
            Origin::Ground { ground, stream } => {
                let ptr = slice.device_ptr(stream).0;
                // The range is the ground's: the view goes without a free.
                slice.leak();
                ground.give_back(ptr, stream);
            }
            Origin::Pool => drop(slice),
        }
    }
}

impl<T: DeviceRepr> std::fmt::Debug for Held<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Held")
            .field("len", &self.slice.len())
            .field("on_ground", &self.on_ground())
            .finish()
    }
}
