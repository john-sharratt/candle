//! A model load's scratch: the top of the span, lent to the load's transients
//! until the dense block is locked (`docs/driver_model.md` §4).
//!
//! The span is one allocation that never grows. A load reads some tensors to
//! the card whole — a projection's source while it is repacked, beside the
//! repack's two bands — and those bytes used to be left outside the span, as
//! driver ground, for the life of the process: the load headroom. But during a
//! load the span's right end belongs to nobody. The dense block grows from the
//! left, and the expert zone is placed from the right only once the block is
//! locked. So the span is claimed whole, and its top `headroom` bytes are
//! opened here for the load's transients, then closed when the load freezes the
//! block — before a weight side is written there.
//!
//! **Only what [`transient`] scopes is served.** A load also makes allocations
//! that outlive it — a weight kept as read, a table built once — and one of
//! those standing in the scratch when the expert zone is placed would be
//! overwritten by expert slots, silently. So the scratch serves an allocation
//! only on a thread inside [`transient`], on the stream it was opened for, and
//! [`close`] refuses while anything it served is still live. A request it
//! cannot hold goes to the driver, as every allocation outside the scope does.
//!
//! The same stream is what makes reuse safe without fences: a range given back
//! is taken again only by work queued after the drop on that stream, and the
//! slice's own drop has already waited on its read and write events.

mod extents;

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use extents::Extents;

use crate::Result;

/// Every take is a whole number of these, at an offset aligned to it.
const ALIGN: usize = 256;

/// One device's open scratch.
struct Scratch {
    ordinal: usize,
    /// The `CUstream` it serves, by address.
    stream: usize,
    base: u64,
    extents: Extents,
    /// Live takes: address → bytes taken.
    live: HashMap<u64, usize>,
    used: usize,
    peak: usize,
    /// Bytes asked for inside a scope that did not fit and went to the driver.
    spilled: usize,
}

/// What a scratch did, reported when it closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchReport {
    /// The scratch's size.
    pub bytes: usize,
    /// The most it held at once.
    pub peak: usize,
    /// Bytes a scoped request asked for that it could not hold.
    pub spilled: usize,
}

/// Whether any scratch is open: the fast path every allocation and free checks.
static ANY_OPEN: AtomicBool = AtomicBool::new(false);

static OPEN: Mutex<Vec<Scratch>> = Mutex::new(Vec::new());

thread_local! {
    /// How deep this thread is in [`transient`].
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn open_scratches() -> std::sync::MutexGuard<'static, Vec<Scratch>> {
    OPEN.lock().unwrap_or_else(|e| e.into_inner())
}

/// Open `len` bytes at `base` on device `ordinal` as its load's scratch,
/// serving allocations on `stream` (a `CUstream`, by address).
pub fn open(ordinal: usize, stream: usize, base: u64, len: usize) -> Result<()> {
    let mut open = open_scratches();
    if open.iter().any(|s| s.ordinal == ordinal) {
        crate::bail!("load scratch: device {ordinal} already has one open");
    }
    open.push(Scratch {
        ordinal,
        stream,
        base,
        extents: Extents::new(len),
        live: HashMap::new(),
        used: 0,
        peak: 0,
        spilled: 0,
    });
    ANY_OPEN.store(true, Ordering::Release);
    Ok(())
}

/// Close device `ordinal`'s scratch, so its ground can be written by whoever
/// owns it next. `Ok(None)` when none is open.
///
/// Refused while an allocation it served is still live: that allocation would
/// be overwritten by the next owner of the ground. The scratch then stays open,
/// and the caller must not hand the ground on.
pub fn close(ordinal: usize) -> Result<Option<ScratchReport>> {
    let mut open = open_scratches();
    let Some(at) = open.iter().position(|s| s.ordinal == ordinal) else {
        return Ok(None);
    };
    let s = &open[at];
    if !s.live.is_empty() || !s.extents.is_whole() {
        crate::bail!(
            "load scratch: device {ordinal} still holds {} B in {} allocation(s) a load \
             scoped as transient; they outlive the load, and the ground they stand on is the \
             weight side's next (docs/driver_model.md §4)",
            s.used,
            s.live.len()
        );
    }
    let s = open.swap_remove(at);
    if open.is_empty() {
        ANY_OPEN.store(false, Ordering::Release);
    }
    Ok(Some(ScratchReport {
        bytes: s.extents.capacity(),
        peak: s.peak,
        spilled: s.spilled,
    }))
}

/// Whether device `ordinal` has a scratch open.
pub fn is_open(ordinal: usize) -> bool {
    ANY_OPEN.load(Ordering::Acquire) && open_scratches().iter().any(|s| s.ordinal == ordinal)
}

/// Run `f` with the device allocations it makes on this thread served from
/// its device's load scratch, where one is open and holds them.
///
/// For a load's transients only — what is freed before the load ends. An
/// allocation `f` makes that outlives the load makes [`close`] refuse.
pub fn transient<R>(f: impl FnOnce() -> R) -> R {
    DEPTH.with(|d| d.set(d.get() + 1));
    struct Leave;
    impl Drop for Leave {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get() - 1));
        }
    }
    let _leave = Leave;
    f()
}

/// Serve `bytes` for `stream` on device `ordinal`, when this thread is inside
/// [`transient`] and that device's scratch serves `stream` and holds them.
pub(crate) fn serve(ordinal: usize, stream: usize, bytes: usize) -> Option<u64> {
    if !ANY_OPEN.load(Ordering::Acquire) || DEPTH.with(Cell::get) == 0 {
        return None;
    }
    let mut open = open_scratches();
    let s = open
        .iter_mut()
        .find(|s| s.ordinal == ordinal && s.stream == stream)?;
    let len = bytes.next_multiple_of(ALIGN);
    let Some(offset) = s.extents.take(len, ALIGN) else {
        s.spilled += bytes;
        return None;
    };
    let ptr = s.base + offset as u64;
    s.live.insert(ptr, len);
    s.used += len;
    s.peak = s.peak.max(s.used);
    Some(ptr)
}

/// Take `ptr` back if a scratch served it: `true`, and the driver frees nothing.
pub(crate) fn take_back(ptr: u64) -> bool {
    if !ANY_OPEN.load(Ordering::Acquire) {
        return false;
    }
    let mut open = open_scratches();
    for s in open.iter_mut() {
        if let Some(len) = s.live.remove(&ptr) {
            s.extents.release((ptr - s.base) as usize, len);
            s.used -= len;
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Device ordinals no other test opens a scratch on.
    const DEVICE: usize = 60;

    /// Inside the scope, on the scratch's stream, a request is served from the
    /// scratch's base and taken back; outside the scope, on another stream or
    /// another device, nothing is.
    #[test]
    fn only_a_scoped_request_on_the_scratch_stream_is_served() -> Result<()> {
        let base = 0x7000_0000u64;
        open(DEVICE, 11, base, 4096)?;
        assert_eq!(serve(DEVICE, 11, 100), None, "outside the scope");
        let (a, b) = transient(|| (serve(DEVICE, 11, 100), serve(DEVICE, 11, 300)));
        assert_eq!(a, Some(base));
        assert_eq!(b, Some(base + 256));
        assert_eq!(transient(|| serve(DEVICE, 12, 100)), None, "another stream");
        assert_eq!(
            transient(|| serve(DEVICE + 1, 11, 100)),
            None,
            "another device"
        );
        assert!(take_back(base));
        assert!(!take_back(base), "taken back once");
        assert!(!take_back(base + 1), "never served");
        assert!(take_back(base + 256));
        assert_eq!(
            close(DEVICE)?,
            Some(ScratchReport {
                bytes: 4096,
                peak: 256 + 512,
                spilled: 0
            })
        );
        assert_eq!(close(DEVICE)?, None, "closed once");
        Ok(())
    }

    /// A request the scratch cannot hold goes to the driver and is counted;
    /// close refuses while a served allocation is live, and the scratch stays
    /// open until it is given back.
    #[test]
    fn close_refuses_while_a_served_allocation_lives() -> Result<()> {
        let base = 0x9000_0000u64;
        open(DEVICE + 2, 7, base, 1024)?;
        let (held, past) = transient(|| (serve(DEVICE + 2, 7, 1024), serve(DEVICE + 2, 7, 1)));
        assert_eq!(held, Some(base));
        assert_eq!(past, None);
        let err = close(DEVICE + 2).expect_err("an allocation is live");
        assert!(err.to_string().contains("1024 B in 1 allocation"), "{err}");
        assert!(take_back(base));
        assert_eq!(
            close(DEVICE + 2)?,
            Some(ScratchReport {
                bytes: 1024,
                peak: 1024,
                spilled: 1
            })
        );
        Ok(())
    }

    /// The scope nests, and leaving it — however `f` ends — stops serving.
    #[test]
    fn the_scope_nests_and_ends_with_its_closure() -> Result<()> {
        let base = 0xb000_0000u64;
        open(DEVICE + 3, 5, base, 4096)?;
        let inner = transient(|| transient(|| serve(DEVICE + 3, 5, 10)));
        assert_eq!(inner, Some(base));
        let _ = std::panic::catch_unwind(|| transient(|| panic!("a load that fails")));
        assert_eq!(
            serve(DEVICE + 3, 5, 10),
            None,
            "the scope ended with the panic"
        );
        assert!(take_back(base));
        close(DEVICE + 3)?;
        Ok(())
    }
}
