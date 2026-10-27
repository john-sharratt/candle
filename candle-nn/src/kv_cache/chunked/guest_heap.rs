//! The guest heap: a guest's activations in its own ground, freed as they die.
//!
//! # Why the guest arena is not enough
//!
//! The guest arena ([`super::bump_arena::open_guest_arena`]) is a bump: a
//! generation's carves are its sum, and memory comes back only when a stage
//! ends and the cursor rewinds. A model written in stages — the image guest's
//! encode, each denoise step, each decode tile — bounds that sum by nesting
//! them. An ONNX graph has no stages: it is one evaluation of tens of thousands
//! of nodes, and a bump over it holds every intermediate at once. BiRefNet at
//! 2048² is 29,105 nodes; IS-Net at 1024² got by only because its sum fitted.
//!
//! Without either, a guest's activations come from the CUDA pool, outside the
//! span — where on a busy WhiteWorld (2026-10-27) the matte guest found 0.6 GB
//! and failed `CUDA_ERROR_OUT_OF_MEMORY` while the ground it had claimed stood
//! unused.
//!
//! # What this is instead
//!
//! First-fit extents ([`super::held_extents::Extents`]) over the run of ground
//! the guest claimed for activations. A carve takes the lowest range that fits;
//! the storage built over it hands it back when it is dropped
//! ([`release_carve`], through [`candle::cuda_backend::wave_provenance::wave_release`]),
//! so the heap holds the guest's **live** set rather than its sum.
//!
//! **A release is fenced**, exactly as held ground's is: the range is reused
//! only once an event recorded on the guest's stream at release has fired, so a
//! launch queued before the drop still reads its bytes.
//!
//! **Only an exact carve is released.** A guest's weights carry a guest ticket
//! too (`varground::guest_origin`) but live in its ground, not here; a release
//! of any address this heap did not carve is not its business and is ignored.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use candle::cuda_backend::cudarc::driver::{sys, CudaEvent, CudaStream};
use candle::cuda_backend::wave_provenance::{
    install_wave_release, SpanState, WaveCarve, WaveTicket, GUEST_ARENA,
};

use super::held_extents::{Extents, Taken};

/// What one heap did over its life: its peak, for sizing the claim next time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuestHeapStats {
    pub capacity: usize,
    /// The most it held at once, alignment padding included.
    pub peak: usize,
    /// Carves it served.
    pub carves: u64,
}

/// One contiguous stretch of the guest's ground, sub-allocated. A carve never
/// crosses from one into the next: between two runs stands another tenant.
struct Run {
    base: u64,
    extents: Extents,
}

struct Heap {
    stream: Arc<CudaStream>,
    runs: Vec<Run>,
    /// The run and range behind each live carve, by the address handed out.
    live: HashMap<u64, (usize, Taken)>,
    /// Ranges given back whose fence may not have fired.
    pending: Vec<(CudaEvent, usize, Taken)>,
    peak: usize,
    carves: u64,
}

impl Heap {
    fn used(&self) -> usize {
        self.runs.iter().map(|r| r.extents.used()).sum()
    }

    fn capacity(&self) -> usize {
        self.runs.iter().map(|r| r.extents.capacity()).sum()
    }

    /// Return every pending range whose fence has fired; with `wait`, the
    /// oldest is waited for first, so at least one comes back.
    fn reclaim(&mut self, wait: bool) {
        if wait {
            if let Some((event, _, _)) = self.pending.first() {
                // A failed wait leaves the range pending rather than reusing
                // memory a launch may still read.
                let _ = event.synchronize();
            }
        }
        let mut i = 0;
        while i < self.pending.len() {
            let fired = unsafe { sys::cuEventQuery(self.pending[i].0.cu_event()) }
                == sys::CUresult::CUDA_SUCCESS;
            if fired {
                let (_, run, t) = self.pending.swap_remove(i);
                self.runs[run].extents.release(t.start, t.len);
            } else {
                i += 1;
            }
        }
    }

    /// The lowest range that fits, in the first run that has one.
    fn take(&mut self, bytes: usize, align: usize) -> Option<u64> {
        self.reclaim(false);
        let (run, taken) = loop {
            let found = self
                .runs
                .iter_mut()
                .enumerate()
                .find_map(|(i, r)| r.extents.take(bytes, align).map(|t| (i, t)));
            if let Some(found) = found {
                break found;
            }
            if self.pending.is_empty() {
                return None;
            }
            // Short only while frees wait on their fences: wait for the oldest.
            self.reclaim(true);
        };
        let ptr = self.runs[run].base + taken.at as u64;
        self.live.insert(ptr, (run, taken));
        self.peak = self.peak.max(self.used());
        self.carves += 1;
        Some(ptr)
    }

    fn release(&mut self, ptr: u64) {
        let Some((run, taken)) = self.live.remove(&ptr) else {
            return;
        };
        match self.stream.record_event(None) {
            Ok(event) => self.pending.push((event, run, taken)),
            // No fence to wait behind: wait for the stream itself rather than
            // reuse bytes a queued launch may still read.
            Err(_) => {
                let _ = self.stream.synchronize();
                self.runs[run].extents.release(taken.start, taken.len);
            }
        }
    }
}

fn heaps() -> &'static Mutex<HashMap<usize, Heap>> {
    static HEAPS: OnceLock<Mutex<HashMap<usize, Heap>>> = OnceLock::new();
    HEAPS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether any heap is open, so a release on the decode path costs one load.
static OPEN: AtomicBool = AtomicBool::new(false);

/// Open a heap over `runs` — `(base, bytes)` stretches of the guest's own
/// ground — for the device `stream` is on. While it is open, a guest ticket on
/// that device carves here rather than from the guest arena.
pub fn open_guest_heap(stream: &Arc<CudaStream>, runs: &[(u64, usize)]) {
    install_wave_release(release_carve);
    super::bump_arena::install_resolver();
    let heap = Heap {
        stream: Arc::clone(stream),
        runs: runs
            .iter()
            .map(|&(base, bytes)| Run {
                base,
                extents: Extents::new(bytes),
            })
            .collect(),
        live: HashMap::new(),
        pending: Vec::new(),
        peak: 0,
        carves: 0,
    };
    let mut map = heaps().lock().unwrap();
    map.insert(stream.context().ordinal(), heap);
    OPEN.store(true, Ordering::Release);
}

/// Close the device's heap and say what it did. **Before the guest's ground
/// goes back**: once the regions are the KV side's again a carve from them would
/// alias attention state. Waits on every pending fence, so nothing queued still
/// reads the ground when it changes hands.
pub fn close_guest_heap(stream: &Arc<CudaStream>) -> Option<GuestHeapStats> {
    let mut map = heaps().lock().unwrap();
    let mut heap = map.remove(&stream.context().ordinal())?;
    OPEN.store(!map.is_empty(), Ordering::Release);
    drop(map);
    let _ = heap.stream.synchronize();
    heap.reclaim(false);
    Some(GuestHeapStats {
        capacity: heap.capacity(),
        peak: heap.peak,
        carves: heap.carves,
    })
}

/// A guest ticket's carve, when its device has a heap open; `None` to let the
/// guest arena answer.
pub(crate) fn carve(ticket: WaveTicket, bytes: usize, align: usize) -> Option<WaveCarve> {
    if !OPEN.load(Ordering::Acquire) {
        return None;
    }
    let Ok(mut map) = heaps().lock() else {
        return Some(WaveCarve::Closed);
    };
    let heap = map.get_mut(&(ticket.domain as usize))?;
    Some(match heap.take(bytes, align) {
        Some(ptr) => WaveCarve::Carved(ptr),
        None => WaveCarve::Exhausted(SpanState {
            cursor: heap.used(),
            capacity: heap.capacity(),
        }),
    })
}

/// The release hook: a storage over `ptr` with a wave ticket has been dropped.
fn release_carve(ticket: WaveTicket, ptr: u64) {
    if !OPEN.load(Ordering::Acquire) || ticket.arena != GUEST_ARENA {
        return;
    }
    if let Ok(mut map) = heaps().lock() {
        if let Some(heap) = map.get_mut(&(ticket.domain as usize)) {
            heap.release(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::gpu_test_lock::gpu_serial;
    use super::*;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::wave_provenance::{wave_alloc, wave_release, LeaseOrigin};
    use candle::{DType, Device, Tensor};

    const MIB: usize = 1 << 20;

    fn carve_tensor(ticket: WaveTicket, device: &Device, bytes: usize) -> Option<(u64, Tensor)> {
        match wave_alloc(ticket, bytes, 256) {
            WaveCarve::Carved(ptr) => {
                // SAFETY: carved from the heap just now, and dropped before it closes.
                let t = unsafe {
                    Tensor::from_leased_cuda_ptr(
                        ptr,
                        DType::U8,
                        bytes,
                        device,
                        LeaseOrigin::Wave(ticket),
                    )
                }
                .unwrap();
                Some((ptr, t))
            }
            _ => None,
        }
    }

    /// **A dropped storage's range comes back, across runs, and only an exact
    /// carve is released.** Two 2 MiB runs: each holds one 2 MiB carve and the
    /// third does not fit anywhere; dropping the first frees run 0, which the
    /// next carve reuses once its fence has fired. A release of an address the
    /// heap never carved — a guest's weights — is ignored.
    #[test]
    fn a_dropped_carve_is_reused_and_a_foreign_release_ignored() {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0).unwrap();
        let cuda = device.as_cuda_device().unwrap();
        let stream = cuda.cuda_stream();
        let ground = stream.alloc_zeros::<u8>(4 * MIB).unwrap();
        let base = ground.device_ptr(&stream).0;
        let ticket = WaveTicket::guest(stream.context().ordinal() as u32);

        open_guest_heap(
            &stream,
            &[(base, 2 * MIB), (base + (2 * MIB) as u64, 2 * MIB)],
        );
        let (a, ta) = carve_tensor(ticket, &device, 2 * MIB).unwrap();
        let (b, tb) = carve_tensor(ticket, &device, 2 * MIB).unwrap();
        assert_eq!((a, b), (base, base + (2 * MIB) as u64), "one carve per run");
        assert!(
            matches!(wave_alloc(ticket, MIB, 256), WaveCarve::Exhausted(_)),
            "a full heap refuses rather than spilling to the pool"
        );

        wave_release(ticket, base + 12345);
        drop(ta);
        device.synchronize().unwrap();
        let (again, tc) = carve_tensor(ticket, &device, 2 * MIB).unwrap();
        assert_eq!(again, a, "the freed range was not reused");

        drop((tb, tc));
        let stats = close_guest_heap(&stream).unwrap();
        assert_eq!(stats.capacity, 4 * MIB);
        assert_eq!(stats.peak, 4 * MIB);
        assert_eq!(stats.carves, 3);
        assert!(
            matches!(wave_alloc(ticket, MIB, 256), WaveCarve::Closed),
            "a closed heap still answered"
        );
        drop(ground);
    }
}
