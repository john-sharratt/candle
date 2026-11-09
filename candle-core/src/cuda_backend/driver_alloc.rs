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
pub(crate) use watch::{at_site, install, report_load_survivors};

/// Nothing is recorded with the feature off.
#[cfg(not(feature = "driver-alloc-panic"))]
#[inline(always)]
pub(crate) fn report_load_survivors(_ordinal: usize) {}

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
    use std::cmp::Reverse;
    use std::collections::HashMap;
    use std::panic::Location;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::sys::CUdeviceptr;
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

    /// A load's allocation still standing: its device, its size, its site, and
    /// the stack that asked — captured unresolved, which is cheap, and resolved
    /// only for the survivors a report prints.
    struct Standing {
        ordinal: usize,
        bytes: usize,
        site: String,
        trace: Backtrace,
    }

    /// The first frame of `trace` in model code — the loader that asked, which
    /// the generic entry point the site names never is.
    fn asked_by(trace: &Backtrace) -> String {
        trace
            .to_string()
            .lines()
            .map(str::trim)
            .find(|l| {
                !l.starts_with("at ")
                    && (l.contains("candle_transformers::") || l.contains("candle_conversation::"))
            })
            .map_or_else(|| "no model frame".to_string(), |l| l.to_string())
    }

    /// Every allocation a load made that has not been freed, by address.
    ///
    /// The exemption a load has is for its transients — the source tensors, the
    /// repack's bands — which are gone when it ends. What is still here when the
    /// last window closes is driver memory the load left behind, outside the
    /// span, for the life of whatever holds it ([`report_load_survivors`]).
    fn standing() -> &'static Mutex<HashMap<CUdeviceptr, Standing>> {
        static STANDING: OnceLock<Mutex<HashMap<CUdeviceptr, Standing>>> = OnceLock::new();
        STANDING.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// The site the allocation now being made on this thread is filed under.
    fn current_site() -> String {
        SITE.with(Cell::get).map_or_else(
            || "an uninstrumented call into cudarc".to_string(),
            |s| s.to_string(),
        )
    }

    /// What device `ordinal`'s loads left standing, the largest site first:
    /// one line per site with its bytes and allocation count, logged when the
    /// device's last [`super::super::LoadWindow`] closes.
    pub(crate) fn report_load_survivors(ordinal: usize) {
        let mut by_site: HashMap<String, (usize, usize)> = HashMap::new();
        for s in standing()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|s| s.ordinal == ordinal)
        {
            let key = format!("{} ← {}", s.site, asked_by(&s.trace));
            let e = by_site.entry(key).or_default();
            e.0 += s.bytes;
            e.1 += 1;
        }
        let mut sites: Vec<(String, (usize, usize))> = by_site.into_iter().collect();
        sites.sort_by_key(|(_, (bytes, _))| Reverse(*bytes));
        let total: usize = sites.iter().map(|(_, (b, _))| b).sum();
        // To stderr, as the refusal beside it panics there: the instrument's
        // output, read by a gate run that installs no subscriber.
        eprintln!(
            "driver allocations a load left standing on device {ordinal}: {} MiB in {} site(s), \
             outside the span",
            total >> 20,
            sites.len()
        );
        for (site, (bytes, count)) in sites {
            eprintln!(
                "  {:>9.1} MiB in {count:>5} allocation(s) at {site}",
                bytes as f64 / (1 << 20) as f64
            );
        }
    }

    /// Refuses every allocation cudarc is asked for outside a model load
    /// ([`super::super::LoadWindow`]), and keeps the census of the ones a load
    /// makes.
    struct Refuse;

    impl AllocWatch for Refuse {
        fn allocated(&self, stream: &Arc<CudaStream>, ptr: CUdeviceptr, bytes: usize) {
            let ordinal = stream.context().ordinal();
            if is_open(ordinal) {
                standing().lock().unwrap_or_else(|e| e.into_inner()).insert(
                    ptr,
                    Standing {
                        ordinal,
                        bytes,
                        site: current_site(),
                        trace: Backtrace::force_capture(),
                    },
                );
            }
        }

        fn freed(&self, ptr: CUdeviceptr) {
            standing()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&ptr);
        }

        fn alloc(&self, stream: &Arc<CudaStream>, bytes: usize) {
            let ordinal = stream.context().ordinal();
            if is_open(ordinal) {
                return;
            }
            let site = current_site();
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
