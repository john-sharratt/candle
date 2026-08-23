//! The measurement layer: a physically-honest read of VRAM residency and the
//! probe trait that produces it.
//!
//! The whole VRAM Governor is built on one principle — the *real* free-VRAM
//! measurement is the single source of truth (see `docs/vram_governor_design.md`
//! §1). A [`VramProbe`] is the platform-specific thing that produces a
//! [`VramReading`]; the governor never keeps a virtual `ceiling − Σcommitted`
//! tally that could drift out of sync with reality.

use crate::Result;

/// Which backend produced a [`VramReading`] — for logging and test assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeKind {
    /// Windows/WDDM: DXGI per-process `Budget − CurrentUsage` (the authoritative
    /// residency signal on a virtualized display driver).
    Dxgi,
    /// Linux / fallback: `cuMemGetInfo` free (accurate where the driver owns the
    /// framebuffer directly).
    Cuda,
    /// Test double.
    Fake,
}

/// A physically-honest snapshot of VRAM residency for one GPU, in bytes.
///
/// `headroom` is the load-bearing field: how many more bytes *this process* can
/// make resident before the OS pages us (WDDM) or the framebuffer is full
/// (Linux). It is a live measurement, never a running tally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramReading {
    /// Bytes we may still make resident before paging / OOM.
    pub headroom: u64,
    /// Total device VRAM.
    pub total: u64,
    /// The backend that produced this reading.
    pub source: ProbeKind,
}

impl VramReading {
    pub fn new(headroom: u64, total: u64, source: ProbeKind) -> Self {
        Self {
            headroom,
            total,
            source,
        }
    }
}

/// A platform backend that measures real free VRAM.
///
/// Implementations must be cheap enough to call on a per-wave cadence (µs–ms).
/// `read` returns the current honest headroom; `budget_change_event` optionally
/// exposes an OS push-notification (Windows) so the governor can re-measure the
/// instant another process perturbs our budget, instead of only polling.
pub trait VramProbe: Send + Sync {
    /// Take one honest reading.
    fn read(&self) -> Result<VramReading>;

    /// A handle the caller can block on that fires when our VRAM budget changes.
    /// `None` on platforms without push notification (Linux, tests) — the
    /// governor falls back to periodic polling there.
    fn budget_change_event(&self) -> Option<Box<dyn BudgetWatchHandle>> {
        None
    }
}

/// An OS handle that becomes signalled when this process's VRAM budget changes.
///
/// On Windows the DXGI probe registers a Win32 auto-reset event
/// (`RegisterVideoMemoryBudgetChangeNotificationEvent`) and returns an
/// implementation of this trait; the governor spawns a thread that blocks on
/// [`BudgetWatchHandle::wait`] and re-measures / relieves on each signal. Kept as
/// a trait object so this module carries no platform code.
pub trait BudgetWatchHandle: Send {
    /// Block until the budget-change event fires or `timeout_ms` elapses.
    /// `true` = fired, `false` = timed out.
    fn wait(&self, timeout_ms: u32) -> bool;
}

/// A scripted probe for unit tests: `headroom` is a shared cell that relief
/// closures mutate, so the ladder terminates deterministically the way real
/// relief (which raises headroom) would drive it.
#[cfg(any(test, feature = "vram-test-util"))]
pub mod fake {
    use super::{ProbeKind, VramProbe, VramReading};
    use crate::Result;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// Shared, mutable headroom used by both [`FakeProbe`] and the test
    /// allocator/relief closures so a test can model "relief raised headroom".
    #[derive(Clone, Default)]
    pub struct FakeVram {
        headroom: Arc<AtomicU64>,
        total: u64,
        reads: Arc<AtomicU64>,
    }

    impl FakeVram {
        pub fn new(headroom: u64, total: u64) -> Self {
            Self {
                headroom: Arc::new(AtomicU64::new(headroom)),
                total,
                reads: Arc::new(AtomicU64::new(0)),
            }
        }
        /// Current scripted headroom.
        pub fn headroom(&self) -> u64 {
            self.headroom.load(Ordering::Relaxed)
        }
        /// Model an allocation consuming `bytes` of headroom (saturating at 0).
        pub fn consume(&self, bytes: u64) {
            let mut cur = self.headroom.load(Ordering::Relaxed);
            loop {
                let next = cur.saturating_sub(bytes);
                match self.headroom.compare_exchange_weak(
                    cur,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(observed) => cur = observed,
                }
            }
        }
        /// Model relief returning `bytes` of headroom.
        pub fn release(&self, bytes: u64) {
            self.headroom.fetch_add(bytes, Ordering::Relaxed);
        }
        pub fn set(&self, headroom: u64) {
            self.headroom.store(headroom, Ordering::Relaxed);
        }
        /// How many times the probe was read (assert polling behaviour).
        pub fn read_count(&self) -> u64 {
            self.reads.load(Ordering::Relaxed)
        }
        pub fn probe(&self) -> FakeProbe {
            FakeProbe {
                vram: self.clone(),
                kind: ProbeKind::Fake,
            }
        }
        /// A probe that reports itself as the given kind — for tests modelling
        /// kind-dependent behaviour (the WDDM wobble margin applies only to
        /// [`ProbeKind::Dxgi`] readings).
        pub fn probe_as(&self, kind: ProbeKind) -> FakeProbe {
            FakeProbe {
                vram: self.clone(),
                kind,
            }
        }
    }

    pub struct FakeProbe {
        vram: FakeVram,
        kind: ProbeKind,
    }

    impl VramProbe for FakeProbe {
        fn read(&self) -> Result<VramReading> {
            self.vram.reads.fetch_add(1, Ordering::Relaxed);
            Ok(VramReading::new(
                self.vram.headroom(),
                self.vram.total,
                self.kind,
            ))
        }
    }
}
