//! Every device allocation cudarc makes, named — and, behind the
//! `driver-alloc-panic` feature, refused (`docs/driver_model.md` §4).
//!
//! The engine's memory is its span: the dense block, the KV regions, the wave
//! transient tier, the expert zone, the span tenants' arenas. An allocation
//! cudarc makes is the driver's, outside all of it — a pool allocation where the
//! card has memory pools, a synchronous `cuMemAlloc` (and a stream synchronise
//! on its free) where it has none, as under TCC. Each one is a site still to be
//! moved onto the arena its lifetime belongs to.
//!
//! A model load is exempt: it runs once, with nothing recording, and goes to the
//! driver while its [`super::LoadWindow`] stands. The feature turns every other
//! one into a panic at the moment it is made, with the
//! `#[track_caller]` site of the instrumented entry point that asked
//! ([`at_site`]) and a backtrace, so a run names the next site to port instead
//! of a number to account for. Off — the default — cudarc allocates as it always
//! did, and [`at_site`] compiles to the closure it wraps.

#[cfg(not(feature = "driver-alloc-panic"))]
use std::panic::Location;

#[cfg(feature = "driver-alloc-panic")]
pub(crate) use watch::{at_site, install};

/// Run `f`. With the feature off nothing is filed: this is the closure.
#[cfg(not(feature = "driver-alloc-panic"))]
#[inline(always)]
pub(crate) fn at_site<R>(_site: &'static Location<'static>, f: impl FnOnce() -> R) -> R {
    f()
}

/// Nothing to install with the feature off.
#[cfg(not(feature = "driver-alloc-panic"))]
#[inline(always)]
pub(crate) fn install() {}

#[cfg(feature = "driver-alloc-panic")]
mod watch {
    use std::backtrace::Backtrace;
    use std::cell::Cell;
    use std::panic::Location;
    use std::sync::Arc;

    use cudarc::driver::{install_alloc_watch, AllocWatch, CudaStream};

    use super::super::held::ground_for;
    use super::super::load_window::is_open;

    thread_local! {
        /// The caller the allocation now being made on this thread is filed under.
        static SITE: Cell<Option<&'static Location<'static>>> = const { Cell::new(None) };
    }

    /// Run `f`, filing the device allocations it makes under `site` — the
    /// `#[track_caller]` location of the instrumented entry point that called
    /// it, which the watch inside cudarc cannot see.
    pub(crate) fn at_site<R>(site: &'static Location<'static>, f: impl FnOnce() -> R) -> R {
        let before = SITE.with(|s| s.replace(Some(site)));
        let out = f();
        SITE.with(|s| s.set(before));
        out
    }

    /// Refuses every allocation cudarc is asked for outside a model load
    /// ([`super::super::LoadWindow`]).
    struct Refuse;

    impl AllocWatch for Refuse {
        fn alloc(&self, stream: &Arc<CudaStream>, bytes: usize) {
            let ordinal = stream.context().ordinal();
            if is_open(ordinal) {
                return;
            }
            let site = SITE.with(Cell::get).map_or_else(
                || "an uninstrumented call into cudarc".to_string(),
                |s| s.to_string(),
            );
            let when = if ground_for(ordinal).is_some() {
                "with the span claimed"
            } else {
                "before the span exists"
            };
            panic!(
                "driver allocation of {bytes} B on device {ordinal} {when}, outside a model \
                 load, at {site}: every runtime device allocation belongs to an arena of the \
                 span (docs/driver_model.md §4)\n{}",
                Backtrace::force_capture()
            );
        }
    }

    /// Install the watch for the process. Idempotent: every device is created
    /// through here, and the first install is the one that stands.
    pub(crate) fn install() {
        install_alloc_watch(Arc::new(Refuse));
    }
}
