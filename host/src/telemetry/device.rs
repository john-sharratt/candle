//! The card and the host, as they actually are.
//!
//! Read through NVML — the same interface `nvidia-smi` uses — which links
//! against the installed driver at runtime and needs no CUDA toolkit. Reaching
//! for `candle` to answer "how much VRAM is free" would have turned a
//! thirty-second daemon build into a twenty-minute one.
//!
//! # Absent is a normal answer
//!
//! No driver, no card, a machine that has never had NVIDIA hardware, a card
//! busy enough to refuse a query — every one of these is ordinary, and none is
//! a reason for `/v1/telemetry` to fail. They all produce `None`, which the
//! console renders as *not measured*. That is deliberately distinct from a
//! reported zero: "no card here" and "the card is idle" are different facts and
//! the difference is exactly what somebody reading this page needs.

use nvml_wrapper::Nvml;
use serde::Serialize;
use sysinfo::System;

/// What the card is, and what is on it.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Gpu {
    pub name: Option<String>,
    pub compute_cap: Option<String>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
}

/// Card memory, in MiB. `None` throughout when there is no card to ask.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Vram {
    pub total_mib: Option<u64>,
    pub used_mib: Option<u64>,
    pub free_mib: Option<u64>,
    /// How the *engine* is spending it. Absent until there is an engine — the
    /// driver can say how much of the card is in use, but not what for.
    pub weights_mib: Option<u64>,
    pub kv_mib: Option<u64>,
    pub image_mib: Option<u64>,
}

/// The machine this daemon is on.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Host {
    pub total_mib: Option<u64>,
    pub free_mib: Option<u64>,
    /// This process, not the machine — the number an operator wants when the
    /// question is whether *the daemon* is the thing eating memory.
    pub rss_mib: Option<u64>,
}

/// Holds the NVML handle open for the life of the daemon.
///
/// Initialising NVML is not free and doing it per request would put that cost
/// on a page that polls every four seconds. A handle that failed to open stays
/// failed: a driver does not appear underneath a running process, and retrying
/// forever would log the same failure every four seconds for as long as the
/// daemon lives.
pub struct Devices {
    nvml: Option<Nvml>,
}

impl Devices {
    pub fn open() -> Self {
        match Nvml::init() {
            Ok(nvml) => {
                tracing::info!("telemetry: NVML available — card metrics are live");
                Self { nvml: Some(nvml) }
            }
            Err(e) => {
                // Info, not warn. A machine without an NVIDIA card is a normal
                // place to run this, and a warning would imply something needs
                // fixing.
                tracing::info!("telemetry: no NVML ({e}) — card metrics unavailable");
                Self { nvml: None }
            }
        }
    }

    /// Device 0. The engine is single-card, so there is one to report; when it
    /// is not, this becomes the place that changes.
    pub fn sample_gpu(&self) -> (Gpu, Vram) {
        let Some(nvml) = &self.nvml else {
            return Default::default();
        };
        let Ok(dev) = nvml.device_by_index(0) else {
            return Default::default();
        };

        let gpu = Gpu {
            name: dev.name().ok(),
            compute_cap: dev
                .cuda_compute_capability()
                .ok()
                .map(|c| format!("{}.{}", c.major, c.minor)),
            // The *current* link, not the maximum the card could negotiate: a
            // card sitting in a slower slot is a thing worth seeing, and it is
            // invisible if the capability is reported instead.
            pcie_gen: dev.current_pcie_link_gen().ok(),
            pcie_width: dev.current_pcie_link_width().ok(),
        };

        let mib = |b: u64| b / (1024 * 1024);
        let vram = match dev.memory_info() {
            Ok(m) => Vram {
                total_mib: Some(mib(m.total)),
                used_mib: Some(mib(m.used)),
                free_mib: Some(mib(m.free)),
                ..Default::default()
            },
            Err(_) => Default::default(),
        };
        (gpu, vram)
    }
}

/// Host memory and this process's footprint.
///
/// A fresh `System` each call, refreshed only for what is asked: the struct
/// otherwise enumerates every process on the machine, which is a great deal of
/// work to do four times a minute in order to read two numbers.
pub fn sample_host() -> Host {
    let mut sys = System::new();
    sys.refresh_memory();

    let mib = |b: u64| b / (1024 * 1024);
    let rss = {
        let pid = sysinfo::get_current_pid().ok();
        pid.and_then(|pid| {
            sys.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                sysinfo::ProcessRefreshKind::new().with_memory(),
            );
            sys.process(pid).map(|p| mib(p.memory()))
        })
    };

    Host {
        total_mib: Some(mib(sys.total_memory())),
        free_mib: Some(mib(sys.available_memory())),
        rss_mib: rss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the whole module: a machine with no card still answers.
    ///
    /// This runs on CI and on laptops. If absence were an error rather than a
    /// value, `/v1/telemetry` would 500 wherever there is no NVIDIA driver —
    /// which is most places that will ever build this.
    #[test]
    fn a_machine_without_a_card_still_reports() {
        let d = Devices::open();
        let (gpu, vram) = d.sample_gpu();
        // Either everything is present or everything is absent; a half-filled
        // card report would be a lie about what was measured.
        if gpu.name.is_some() {
            assert!(vram.total_mib.is_some(), "a named card reported no memory");
            assert!(vram.total_mib.unwrap() > 0);
        } else {
            assert!(vram.total_mib.is_none());
        }
        // What the engine spends VRAM on is never known here, card or no card.
        assert!(vram.weights_mib.is_none() && vram.kv_mib.is_none());
    }

    #[test]
    fn the_host_is_always_measurable() {
        let h = sample_host();
        let total = h.total_mib.expect("a machine knows its own memory");
        assert!(total > 0);
        assert!(h.free_mib.unwrap() <= total);
        // The daemon is a process on it, so its footprint is smaller than the
        // machine's — a sanity check on the units, which are easy to mix up.
        if let Some(rss) = h.rss_mib {
            assert!(rss <= total, "rss {rss} MiB exceeds host {total} MiB");
        }
    }
}
