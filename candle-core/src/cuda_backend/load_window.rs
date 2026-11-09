//! The window a model load runs in, per device.
//!
//! A load runs once, with nothing recording, so a driver allocation there costs
//! a `cuMemAlloc` and nothing more: the load's sources, its repack bands and the
//! layer pack's build go to the driver, and what the load keeps outside the span
//! is the "outside the span" figure the span report prints. Outside every load,
//! a driver allocation is a runtime one — a driver call per allocation and, with
//! no memory pools (TCC), a stream synchronise per free — and that is what the
//! `driver-alloc-panic` feature refuses (`docs/driver_model.md` §4).
//!
//! A window is counted, not flagged: a process that loads a second model opens a
//! second window, and the device is in a load while any stands.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::driver_alloc::report_load_survivors;

/// Devices a window can be opened on: one counter each.
const MAX_DEVICES: usize = 64;

static OPEN: [AtomicUsize; MAX_DEVICES] = [const { AtomicUsize::new(0) }; MAX_DEVICES];

/// A model load on one device, open until dropped.
#[must_use = "the window closes when the guard drops"]
pub struct LoadWindow {
    ordinal: usize,
}

impl LoadWindow {
    /// Open a load window on device `ordinal`.
    ///
    /// # Panics
    ///
    /// On an ordinal past the [`MAX_DEVICES`] the counters cover.
    pub fn open(ordinal: usize) -> Self {
        assert!(
            ordinal < MAX_DEVICES,
            "load window: device {ordinal} is past the {MAX_DEVICES} devices counted"
        );
        OPEN[ordinal].fetch_add(1, Ordering::AcqRel);
        Self { ordinal }
    }
}

impl Drop for LoadWindow {
    fn drop(&mut self) {
        // The last window on the device closing is the load ending: with the
        // `driver-alloc-panic` census on, what it left standing outside the span
        // is reported now, while the sites are still the load's.
        if OPEN[self.ordinal].fetch_sub(1, Ordering::AcqRel) == 1 {
            report_load_survivors(self.ordinal);
        }
    }
}

/// Whether a load window stands on device `ordinal`.
#[cfg_attr(not(feature = "driver-alloc-panic"), allow(dead_code))]
pub(crate) fn is_open(ordinal: usize) -> bool {
    OPEN.get(ordinal)
        .is_some_and(|n| n.load(Ordering::Acquire) > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Windows are counted: the device is in a load while any stands, and not
    /// once the last one drops.
    #[test]
    fn a_device_is_in_a_load_while_any_window_stands() {
        // A device no other test opens a window on.
        let ordinal = MAX_DEVICES - 1;
        assert!(!is_open(ordinal));
        let first = LoadWindow::open(ordinal);
        let second = LoadWindow::open(ordinal);
        drop(first);
        assert!(is_open(ordinal), "the second window still stands");
        drop(second);
        assert!(!is_open(ordinal));
        assert!(
            !is_open(MAX_DEVICES),
            "past the counters is never in a load"
        );
    }
}
