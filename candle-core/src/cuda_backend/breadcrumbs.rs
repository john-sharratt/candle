//! Records the recent kernel launches of each thread, so a CUDA fault can be
//! traced to the launch that caused it when its error surfaces later.
//!
//! CUDA reports kernel errors at a later synchronisation point, often on
//! another wrapper or thread, so the last launch is rarely the culprit. Each
//! thread keeps a short ring of its launches, written on the host just before
//! every kernel call, so the kernels themselves are untouched. The error hook
//! dumps that history, and the faulting launch stays visible even when
//! detection drifts a few launches past it. `CUDA_LAUNCH_BLOCKING=1` gives
//! exact attribution during a debug session.

use super::*;

// ── Kernel breadcrumb ─────────────────────────────────────────────────────────
//
// Written immediately before every kernel FFI call (one thread-local store,
// ~1 ns) — HOST-SIDE, before the launch, so the kernels themselves are never
// touched and there is zero kernel-side perf cost.
//
// CUDA errors are ASYNCHRONOUS: a faulting kernel surfaces its error at a LATER
// synchronization point (the next launch's error check, a device sync, a memcpy)
// — often on a DIFFERENT wrapper, sometimes a different thread. A single
// "last kernel" slot therefore names the wrong kernel much of the time. We keep
// a short RING of recent launches per thread instead, so the error hook can dump
// the recent history and the true culprit is visible even when the point of
// detection drifts a few launches past the point of fault.
//
// For EXACT attribution during a debug session, run with `CUDA_LAUNCH_BLOCKING=1`
// (driver-level: every launch synchronizes, so the error surfaces AT the faulting
// launch and breadcrumb #0 is the culprit), and/or `compute-sanitizer` — the
// kernels are built with `--generate-line-info` (see `candle-kernels/build_utils`),
// so the sanitizer reports the exact `.cu` file and line of the out-of-bounds
// access at no runtime cost.
//
// Usage: call `cuda_breadcrumb!("run_foo")` at the top of each wrapper that
// dispatches to a kernel.  The macro captures `file!()` / `line!()` at the
// actual call site so the recorded location is meaningful.

/// How many recent launches to keep per thread. Small — the fault is nearly
/// always within the last handful of launches ahead of the detection point.
pub(super) const KERNEL_RING_LEN: usize = 16;

/// One thread's recent launches, newest at `pos - 1`.
pub(super) struct KernelRing {
    pub(super) slots: [(&'static str, &'static str, u32); KERNEL_RING_LEN],
    /// Index of the NEXT slot to write (ring is `[pos-1, pos-2, …]` newest→oldest).
    pub(super) pos: usize,
}

impl KernelRing {
    pub(super) fn render(&self) -> Vec<String> {
        let mut out = Vec::new();
        for k in 0..KERNEL_RING_LEN {
            let i = (self.pos + KERNEL_RING_LEN - 1 - k) % KERNEL_RING_LEN;
            let (name, file, line) = self.slots[i];
            if name.is_empty() {
                continue;
            }
            out.push(format!("    #{k} '{name}' ({file}:{line})"));
        }
        out
    }
}

/// Every thread's ring, by thread name, so a fault report can show the threads
/// that did **not** notice it. The thread that reaches a sync first is rarely
/// the one that launched the faulting kernel: a background thread that syncs
/// often reports the fault, and the launcher's history is the one that names it.
pub(super) static KERNEL_RINGS: Mutex<Vec<(String, Weak<Mutex<KernelRing>>)>> =
    Mutex::new(Vec::new());

thread_local! {
    static KERNEL_RING: Arc<Mutex<KernelRing>> = {
        let ring = Arc::new(Mutex::new(KernelRing {
            slots: [("", "", 0); KERNEL_RING_LEN],
            pos: 0,
        }));
        let name = std::thread::current()
            .name()
            .map_or_else(|| format!("{:?}", std::thread::current().id()), str::to_string);
        let mut rings = KERNEL_RINGS.lock().unwrap_or_else(|p| p.into_inner());
        rings.retain(|(_, w)| w.strong_count() > 0);
        rings.push((name, Arc::downgrade(&ring)));
        ring
    };
}

/// Record a breadcrumb for the current thread's latest kernel launch.
/// Called via the `cuda_breadcrumb!` macro before each kernel FFI call.
#[inline(always)]
pub fn set_kernel_breadcrumb(name: &'static str, file: &'static str, line: u32) {
    KERNEL_RING.with(|r| {
        let mut ring = r.lock().unwrap_or_else(|p| p.into_inner());
        let pos = ring.pos;
        ring.slots[pos] = (name, file, line);
        ring.pos = (pos + 1) % KERNEL_RING_LEN;
    });
}

/// Dump the recent kernel-launch breadcrumbs on this thread, newest first.
/// Because CUDA errors are asynchronous, the faulting kernel is usually `#0`
/// but may be a few entries back — the history is what makes that visible.
/// Read from the panic hook / CUDA error sites when a `DriverError` surfaces.
pub fn last_cuda_kernel_launch() -> String {
    let out = KERNEL_RING.with(|r| r.lock().unwrap_or_else(|p| p.into_inner()).render());
    if out.is_empty() {
        "(no kernels recorded on this thread)".to_string()
    } else {
        format!(
            "recent CUDA kernel launches on this thread, newest first — async \
             error, fault is usually #0 but can be a few back:\n{}",
            out.join("\n")
        )
    }
}

/// Every live thread's recent launches, newest first per thread — the record a
/// sticky fault keeps, because the error surfaces on whichever thread
/// synchronises next and that is usually not the thread that launched the
/// faulting kernel.
pub fn every_thread_kernel_launches() -> String {
    let rings = KERNEL_RINGS.lock().unwrap_or_else(|p| p.into_inner());
    let mut out = vec![
        "recent CUDA kernel launches on every thread, newest first — async error, the \
         faulting kernel is near the top of one of these:"
            .to_string(),
    ];
    for (name, weak) in rings.iter() {
        let Some(ring) = weak.upgrade() else { continue };
        let lines = ring.lock().unwrap_or_else(|p| p.into_inner()).render();
        if lines.is_empty() {
            continue;
        }
        out.push(format!("  thread '{name}':"));
        out.extend(lines);
    }
    out.join("\n")
}
