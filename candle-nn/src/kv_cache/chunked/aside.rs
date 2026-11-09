//! An aside: a bump on held ground for device work that runs between forwards.
//!
//! # What runs here
//!
//! An op allocates its output from its operand's arena (`wave_provenance`). A
//! forward's values carry its wave's ticket, so everything derived from them
//! during the forward lands in the forward's tier. What runs *after* a forward
//! has none: a sampler picking from logits copied out of the tier, a check that
//! the pick scored finite, a harness comparing rows across configurations. Its
//! operands are owned, an op on an owned operand has no arena to inherit, and its
//! output went to the driver — one `cuMemAlloc` per op and, on a card with no
//! memory pools (TCC), a stream synchronise per free.
//!
//! An aside gives that work an arena. [`aside`] takes one range of held ground,
//! seeds every ticketless operand on this thread with the aside's ticket
//! ([`with_ambient`]), and runs the closure: each op's output is a cursor bump in
//! that range. The range goes back as one give-back when the closure is done.
//!
//! # A value that outlives its aside
//!
//! The closure may hand back a tensor carved inside the aside. That is not an
//! error and not a dangling range: the aside counts the carves still standing,
//! and keeps its ground until the last of them is dropped. What the escaped value
//! cannot do is allocate there — an op on it after the aside has closed finds the
//! ticket closed, and its output goes where an owned operand's goes.
//!
//! # Sizing
//!
//! The caller names the capacity, and the aside makes room for it on held
//! ground before it takes — claiming an arena if no free range holds it, which
//! an ordinary take may not do from wherever it is called. An op that overruns
//! the capacity is refused
//! ([`WaveCarve::Exhausted`]), naming the site, exactly as a wave phase is: an
//! aside sized short is a bug to fix where it was sized, not a reason to spill.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::wave_provenance::{
    with_ambient, SpanState, WaveCarve, WaveTicket, ASIDE_ARENA,
};
use candle::cuda_backend::Held;
use candle::{Device, Result};

use super::region_pool::make_held_room;

/// One open — or closed but still referenced — aside.
struct Aside {
    /// The ground the bump stands in. Dropped, and so given back behind the
    /// stream's queued launches, once the aside is closed and no carve stands.
    /// Held for its drop alone; the bump reads `base`.
    _ground: Held<u8>,
    base: u64,
    capacity: usize,
    cursor: usize,
    /// The carves whose storages are still alive, by address.
    live: HashSet<u64>,
    /// Whether the closure is still running: only then does a ticket carve.
    open: bool,
}

/// Every aside by id. Asides are few and short, so one lock serves them all.
fn asides() -> &'static Mutex<HashMap<u64, Aside>> {
    static ASIDES: OnceLock<Mutex<HashMap<u64, Aside>>> = OnceLock::new();
    ASIDES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The id the next aside takes. Never reused, so a ticket from a closed aside
/// cannot name a later one.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Run `f` with every op on a ticketless operand allocating from `capacity`
/// bytes of held ground on `device`.
///
/// Off CUDA, and for a zero capacity, this is `f()`.
#[track_caller]
pub fn aside<R>(device: &Device, capacity: usize, f: impl FnOnce() -> Result<R>) -> Result<R> {
    let Ok(cuda) = device.as_cuda_device() else {
        return f();
    };
    if capacity == 0 {
        return f();
    }
    super::bump_arena::install_resolver();
    super::wave_release::install();
    let stream = cuda.cuda_stream();
    // An aside opens between forwards with no lock held, so it may claim: its
    // range is covered now rather than refused to the driver once.
    make_held_room(cuda, capacity);
    // SAFETY: the bump hands out ranges whose every byte the op that carved
    // them writes before anything reads it, as any arena's.
    let ground = unsafe { Held::<u8>::alloc_on(&stream, capacity)? };
    let base = ground.device_ptr(&stream).0;
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    asides().lock().unwrap().insert(
        id,
        Aside {
            _ground: ground,
            base,
            capacity,
            cursor: 0,
            live: HashSet::new(),
            open: true,
        },
    );
    let ticket = WaveTicket {
        domain: stream.context().ordinal() as u32,
        arena: ASIDE_ARENA,
        epoch: id,
    };
    // Closed on unwind too: a panicking closure must not leave its ticket
    // carving, nor its ground held by an aside nobody will close.
    struct Close(u64);
    impl Drop for Close {
        fn drop(&mut self) {
            close(self.0);
        }
    }
    let _close = Close(id);
    with_ambient(ticket, f)
}

/// Stop `id` carving, and give its ground back if nothing it carved stands.
fn close(id: u64) {
    let mut map = asides().lock().unwrap_or_else(|e| e.into_inner());
    let done = match map.get_mut(&id) {
        Some(a) => {
            a.open = false;
            a.live.is_empty()
        }
        None => false,
    };
    // Removed under the lock and dropped outside it: the give-back takes the
    // ground's own lock.
    let gone = done.then(|| map.remove(&id)).flatten();
    drop(map);
    drop(gone);
}

/// Carve for a ticket naming an aside: the resolver's answer for
/// [`ASIDE_ARENA`].
pub(super) fn carve(ticket: WaveTicket, bytes: usize, align: usize) -> WaveCarve {
    let Ok(mut map) = asides().lock() else {
        return WaveCarve::Closed;
    };
    let Some(a) = map.get_mut(&ticket.epoch).filter(|a| a.open) else {
        return WaveCarve::Closed;
    };
    match bump(a.base, a.capacity, a.cursor, bytes, align) {
        Some((at, cursor)) => {
            a.cursor = cursor;
            let ptr = a.base + at as u64;
            a.live.insert(ptr);
            WaveCarve::Carved(ptr)
        }
        None => WaveCarve::Exhausted(SpanState {
            cursor: a.cursor,
            capacity: a.capacity,
        }),
    }
}

/// A storage over an aside's carve is gone. The last one of a closed aside
/// gives its ground back.
pub(super) fn release(ticket: WaveTicket, ptr: u64) {
    let gone = {
        let Ok(mut map) = asides().lock() else {
            return;
        };
        let Some(a) = map.get_mut(&ticket.epoch) else {
            return;
        };
        a.live.remove(&ptr);
        if !a.open && a.live.is_empty() {
            map.remove(&ticket.epoch)
        } else {
            None
        }
    };
    drop(gone);
}

/// The offset a carve of `bytes` at `align` takes from a bump at `cursor`, and
/// the cursor after it — `None` past `capacity`. The address, not the offset,
/// is what is aligned: held ground hands out 256-aligned ranges, but the bump
/// does not rely on it.
fn bump(
    base: u64,
    capacity: usize,
    cursor: usize,
    bytes: usize,
    align: usize,
) -> Option<(usize, usize)> {
    let align = align.max(1) as u64;
    let at = ((base + cursor as u64).next_multiple_of(align) - base) as usize;
    let end = at.checked_add(bytes.max(1))?;
    (end <= capacity).then_some((at, end))
}

/// Asides open or still referenced, for a test to see that one went back.
#[cfg(test)]
fn standing() -> usize {
    asides().lock().unwrap().len()
}

#[cfg(test)]
mod tests {
    use super::super::gpu_test_lock::gpu_serial;
    use super::*;
    use candle::cuda_backend::wave_provenance::wave_alloc;
    use candle::{DType, Tensor};

    #[test]
    fn a_bump_aligns_the_address_and_refuses_past_capacity() {
        assert_eq!(bump(0x1000, 1024, 0, 100, 256), Some((0, 100)));
        assert_eq!(bump(0x1000, 1024, 100, 100, 256), Some((256, 356)));
        assert_eq!(bump(0x1010, 1024, 0, 16, 256), Some((240, 256)));
        assert_eq!(bump(0x1000, 1024, 900, 200, 1), None);
        assert_eq!(
            bump(0x1000, 1024, 1024, 0, 1),
            None,
            "an empty ask still takes a byte"
        );
    }

    /// **Ops on an owned operand carve from the aside, a carve that escapes keeps
    /// the ground, and the ticket stops carving once the aside closes.**
    #[test]
    fn ops_carve_from_the_aside_and_an_escaped_value_keeps_its_ground() {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0).unwrap();
        let before = standing();
        let owned = Tensor::arange(0f32, 1024., &device).unwrap();
        let (sum, kept, ticket) = aside(&device, 1 << 20, || {
            let doubled = (&owned * 2.0)?;
            let ticket = doubled.wave_ticket().expect("the op carved from the aside");
            assert_eq!(ticket.arena, ASIDE_ARENA);
            let sum = doubled.sum_all()?.to_vec0::<f32>()?;
            Ok((sum, doubled, ticket))
        })
        .unwrap();
        assert_eq!(sum, 1023. * 1024.);
        assert_eq!(standing(), before + 1, "the escaped carve keeps its aside");
        assert!(
            matches!(wave_alloc(ticket, 64, 256), WaveCarve::Closed),
            "a closed aside still carved"
        );
        assert_eq!(kept.dtype(), DType::F32);
        assert_eq!(kept.to_vec1::<f32>().unwrap()[1023], 2046.);
        drop(kept);
        assert_eq!(
            standing(),
            before,
            "the last carve's drop gave the ground back"
        );
    }

    /// An aside sized short refuses the op that overruns it.
    #[test]
    fn an_overrun_is_refused() {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0).unwrap();
        let owned = Tensor::zeros(4096, DType::F32, &device).unwrap();
        let r = aside(&device, 1024, || &owned + 1.0);
        assert!(r.is_err(), "a 16 KiB output fitted a 1 KiB aside");
    }
}
