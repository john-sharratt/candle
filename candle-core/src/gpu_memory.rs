//! Global GPU memory tracker for diagnostics.
//!
//! Provides a lightweight tracking mechanism to understand GPU memory usage
//! at different phases of model execution. Categories include model weights,
//! arena memory (by dtype), and snapshot-based delta tracking.
//!
//! # Usage
//!
//! ```ignore
//! use candle::gpu_memory;
//!
//! // Take a snapshot to record current GPU free memory
//! gpu_memory::snapshot("after_model_load", &device)?;
//!
//! // Register a known allocation category
//! gpu_memory::register("arena/BF16", bytes);
//!
//! // Print the full report
//! gpu_memory::print_report(&device)?;
//! ```

use crate::{Device, Result};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// A single memory snapshot capturing free/total GPU memory at a labeled point.
#[derive(Debug, Clone)]
struct Snapshot {
    label: String,
    free: usize,
    #[allow(dead_code)]
    total: usize,
}

/// Global tracker state.
struct TrackerState {
    /// Chronological snapshots of GPU memory at labeled points.
    snapshots: Vec<Snapshot>,
    /// Registered allocation categories with byte counts.
    allocations: BTreeMap<String, usize>,
}

static TRACKER: Mutex<Option<TrackerState>> = Mutex::new(None);

fn with_tracker<R>(f: impl FnOnce(&mut TrackerState) -> R) -> R {
    let mut guard = TRACKER.lock().unwrap();
    let state = guard.get_or_insert_with(|| TrackerState {
        snapshots: Vec::new(),
        allocations: BTreeMap::new(),
    });
    f(state)
}

/// Take a snapshot of current GPU free/total memory with a label.
///
/// Snapshots are stored chronologically. The delta between consecutive
/// snapshots shows memory consumed by the intervening operations.
pub fn snapshot(label: &str, device: &Device) -> Result<()> {
    let (free, total) = device.mem_get_info()?;
    with_tracker(|s| {
        s.snapshots.push(Snapshot {
            label: label.to_string(),
            free,
            total,
        });
    });
    Ok(())
}

/// Register (or update) a named allocation category with its byte count.
///
/// This is additive — if the same label is registered multiple times,
/// the value is replaced (not accumulated). Use `add()` for accumulation.
pub fn register(label: &str, bytes: usize) {
    with_tracker(|s| {
        s.allocations.insert(label.to_string(), bytes);
    });
}

/// Add bytes to a named allocation category (accumulate).
pub fn add(label: &str, bytes: usize) {
    with_tracker(|s| {
        let entry = s.allocations.entry(label.to_string()).or_insert(0);
        *entry += bytes;
    });
}

/// Clear all tracked data (snapshots and allocations).
pub fn clear() {
    with_tracker(|s| {
        s.snapshots.clear();
        s.allocations.clear();
    });
}

/// Format a byte count as a human-readable string (MiB).
fn fmt_mb(bytes: usize) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// Build and return the full memory report as a formatted string.
///
/// Includes:
/// 1. Current GPU free/total from `device.mem_get_info()`
/// 2. All registered allocation categories
/// 3. Snapshot history with deltas
pub fn format_report(device: &Device) -> Result<String> {
    let (free, total) = device.mem_get_info()?;
    let used = total.saturating_sub(free);

    let mut lines = Vec::new();
    lines.push(String::new());
    lines.push("╔══════════════════════════════════════════════════════════╗".into());
    lines.push("║              GPU MEMORY USAGE REPORT                    ║".into());
    lines.push("╠══════════════════════════════════════════════════════════╣".into());
    lines.push(format!(
        "║  Total GPU memory:  {:>10}                          ║",
        fmt_mb(total)
    ));
    lines.push(format!(
        "║  Used  GPU memory:  {:>10}                          ║",
        fmt_mb(used)
    ));
    lines.push(format!(
        "║  Free  GPU memory:  {:>10}                          ║",
        fmt_mb(free)
    ));

    // Registered allocations
    let allocs: Vec<(String, usize)> =
        with_tracker(|s| s.allocations.iter().map(|(k, v)| (k.clone(), *v)).collect());

    if !allocs.is_empty() {
        lines.push("╠══════════════════════════════════════════════════════════╣".into());
        lines.push("║  Tracked Allocations:                                   ║".into());
        lines.push("║  ┌─────────────────────────────────┬──────────────────┐ ║".into());
        lines.push("║  │ Category                        │ Size             │ ║".into());
        lines.push("║  ├─────────────────────────────────┼──────────────────┤ ║".into());
        let mut tracked_total = 0usize;
        for (label, bytes) in &allocs {
            tracked_total += bytes;
            lines.push(format!("║  │ {:<33} │ {:>16} │ ║", label, fmt_mb(*bytes)));
        }
        lines.push("║  ├─────────────────────────────────┼──────────────────┤ ║".into());
        lines.push(format!(
            "║  │ {:<33} │ {:>16} │ ║",
            "TRACKED TOTAL",
            fmt_mb(tracked_total)
        ));
        let untracked = used.saturating_sub(tracked_total);
        lines.push(format!(
            "║  │ {:<33} │ {:>16} │ ║",
            "UNTRACKED (fragmentation/other)",
            fmt_mb(untracked)
        ));
        lines.push("║  └─────────────────────────────────┴──────────────────┘ ║".into());
    }

    // Snapshots with deltas
    let snapshots: Vec<Snapshot> = with_tracker(|s| s.snapshots.clone());

    if !snapshots.is_empty() {
        lines.push("╠══════════════════════════════════════════════════════════╣".into());
        lines.push("║  Memory Snapshots (chronological):                      ║".into());
        lines.push("║  ┌─────────────────────────┬──────────┬────────────────┐ ║".into());
        lines.push("║  │ Label                   │ Free     │ Delta (used)   │ ║".into());
        lines.push("║  ├─────────────────────────┼──────────┼────────────────┤ ║".into());
        for (i, snap) in snapshots.iter().enumerate() {
            let delta = if i > 0 {
                let prev = &snapshots[i - 1];
                let consumed = prev.free.saturating_sub(snap.free);
                format!("+{}", fmt_mb(consumed))
            } else {
                "—".into()
            };
            lines.push(format!(
                "║  │ {:<25} │ {:>8} │ {:>14} │ ║",
                snap.label,
                fmt_mb(snap.free),
                delta
            ));
        }
        lines.push("║  └─────────────────────────┴──────────┴────────────────┘ ║".into());
    }

    lines.push("╚══════════════════════════════════════════════════════════╝".into());
    lines.push(String::new());

    Ok(lines.join("\n"))
}

/// Print the full memory report to stderr.
pub fn print_report(device: &Device) -> Result<()> {
    let report = format_report(device)?;
    eprintln!("{}", report);
    Ok(())
}
