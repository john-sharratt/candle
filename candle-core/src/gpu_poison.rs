//! Global GPU-context poison flag.
//!
//! A *sticky* GPU fault — a CUDA illegal address, launch failure, misaligned
//! access, device assert, ECC/NVLink uncorrectable — leaves the device context
//! permanently unusable: every later call returns the same error, forever. There
//! is no API to clear it on the same context, and recreating the context
//! in-process is unreliable (especially on WDDM). So instead of letting an
//! endless cascade of identical downstream errors spew until the process is
//! killed by hand, the backend FLAGS the context poisoned on the first such
//! fault. The daemon watches this flag and exits cleanly for a supervisor
//! restart — one root fault, then a fast reboot. The substrate redo log is
//! crash-safe, so the abrupt exit loses nothing durable.
//!
//! The flag lives here in candle-core (not the `cuda`-gated backend) so a host
//! daemon can poll it without a `cuda` cfg; the CUDA backend is the only writer.
//! `OUT_OF_MEMORY` is deliberately NOT treated as sticky — it is recoverable and
//! handled elsewhere (the code_read ingest retry).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

static POISONED: AtomicBool = AtomicBool::new(false);
static ROOT_FAULT: OnceLock<String> = OnceLock::new();

/// Flag the GPU context poisoned. `root` — the fault plus the recent
/// kernel-launch breadcrumb — is evaluated and captured ONCE, on the first
/// poison. Returns `true` iff this call was the transition, so the caller
/// records the root exactly once and later poisons stay quiet.
pub fn poison_gpu(root: impl FnOnce() -> String) -> bool {
    let first = !POISONED.swap(true, Ordering::SeqCst);
    if first {
        let _ = ROOT_FAULT.set(root());
    }
    first
}

/// Whether a sticky GPU fault has poisoned the context.
#[inline]
pub fn is_gpu_poisoned() -> bool {
    POISONED.load(Ordering::Relaxed)
}

/// The captured root fault (error + recent launches) if the context is poisoned.
pub fn root_fault() -> Option<String> {
    ROOT_FAULT.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_poison_captures_root_and_reports_transition() {
        // Fresh statics per test binary; this is the only test touching them.
        assert!(!is_gpu_poisoned());
        assert!(
            poison_gpu(|| "root-A".to_string()),
            "first poison is the transition"
        );
        assert!(is_gpu_poisoned());
        assert_eq!(root_fault().as_deref(), Some("root-A"));
        // A second poison is not the transition and does not overwrite the root.
        assert!(!poison_gpu(|| "root-B".to_string()));
        assert_eq!(root_fault().as_deref(), Some("root-A"));
    }
}
