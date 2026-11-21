//! Keeps a process-wide list of backings, so one can ask the others to release
//! arenas or report memory.
//!
//! When one backing cannot grow its arenas, it can ask every other registered
//! backing to release its empty arenas. The registry holds weak references, so
//! dropped backings are skipped. Releasing an arena is a free-list push, so it is
//! safe wherever a region is claimed. The table printers write fixed-width
//! diagnostics to stderr.

use super::*;

/// Global registry of all ChunkedKvBacking instances for cooperative compaction.
/// When one backing needs to grow arenas but fails (OOM), it can ask others to compact.
pub(super) static BACKING_REGISTRY: Mutex<Vec<Weak<BackingInner>>> = Mutex::new(Vec::new());

pub(in super::super) fn register_backing(inner: &Arc<BackingInner>) {
    if let Ok(mut registry) = BACKING_REGISTRY.lock() {
        // Clean up dead entries while we're here
        registry.retain(|w| w.strong_count() > 0);
        registry.push(Arc::downgrade(inner));
    }
}

/// Collect a combined memory report from ALL registered ChunkedKvBacking instances.
///
/// Returns a Vec of `(backing_index, format_label, arena_count, total_bytes)`.
/// Dead (dropped) backings are skipped.
pub fn global_arena_memory_report() -> Vec<(usize, String, usize, usize)> {
    let mut results = Vec::new();
    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for (layer_idx, weak) in registry.iter().enumerate() {
            if let Some(backing) = weak.upgrade() {
                let backing = ChunkedKvBacking {
                    inner: backing,
                    layer_idx,
                    state: std::sync::Arc::new(std::sync::RwLock::new(
                        super::super::types::BlockTableState::new(layer_idx, 1, 0),
                    )),
                };
                if let Ok(report) = backing.memory_report() {
                    for (label, count, bytes) in report {
                        results.push((layer_idx, label, count, bytes));
                    }
                }
            }
        }
    }
    results
}

/// Release every fully-empty arena across all registered backings, returning
/// their regions to the pool. Returns arenas freed.
///
/// The reactive twin of the per-backing sweep, callable from the region layer
/// itself: the transient-tier placement (`region_pool::place_transient`)
/// measures its footprint against *claimed* regions, and a region whose arena
/// went chunk-empty since the last periodic sweep still reads as claimed —
/// measured on the Llama-2 MHA gate, 238 empty arenas stood between a 4-region
/// tier and its ground. An arena release is a free-list push, so this is safe
/// wherever a claim is.
///
/// Called from `region_pool`, which is CUDA-only — there is no transient tier
/// to place against without a device.
#[cfg(feature = "cuda")]
pub(in super::super) fn global_release_empty_arenas() -> usize {
    let mut freed = 0;
    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for weak in registry.iter() {
            if let Some(inner) = weak.upgrade() {
                // The pool's atomic fast path makes a no-op backing cost one
                // load, so this is callable from every claim without weight.
                if inner.pool.has_reclaimable() {
                    freed += inner.release_empty_arenas().unwrap_or(0);
                }
            }
        }
    }
    freed
}

/// Get the total GPU memory used by all registered arena backings (bytes).
pub fn global_arena_gpu_bytes() -> usize {
    let mut total = 0;
    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for (layer_idx, weak) in registry.iter().enumerate() {
            if let Some(backing) = weak.upgrade() {
                let backing = ChunkedKvBacking {
                    inner: backing,
                    layer_idx,
                    state: std::sync::Arc::new(std::sync::RwLock::new(
                        super::super::types::BlockTableState::new(layer_idx, 1, 0),
                    )),
                };
                if let Ok(bytes) = backing.total_arena_gpu_bytes() {
                    total += bytes;
                }
            }
        }
    }
    total
}

/// Print a detailed per-arena diagnostic table for ALL registered backings to stderr.
///
/// Aggregates arena rows across all live backings into a single table.  Each row
/// is tagged with its backing index (registry slot) — NOT a layer index, since
/// arenas are typically shared across all layers.  Empty backings are omitted.
pub fn global_print_arena_table() {
    use crate::kv_cache::chunked::arena::ArenaRow;
    use std::collections::BTreeMap;

    // Collect rows from all live backings, tagging each with its backing index.
    let mut tagged_rows: Vec<(usize, ArenaRow)> = Vec::new();
    let mut live_backings = 0usize;
    let mut dead_backings = 0usize;

    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for (layer_idx, weak) in registry.iter().enumerate() {
            if let Some(inner) = weak.upgrade() {
                live_backings += 1;
                let backing = ChunkedKvBacking {
                    inner,
                    layer_idx,
                    state: std::sync::Arc::new(std::sync::RwLock::new(
                        super::super::types::BlockTableState::new(layer_idx, 1, 0),
                    )),
                };
                if let Ok(rows) = backing.arena_rows() {
                    if !rows.is_empty() {
                        for row in rows {
                            tagged_rows.push((layer_idx, row));
                        }
                    }
                }
            } else {
                dead_backings += 1;
            }
        }
    }

    if live_backings == 0 {
        eprintln!(
            "\n[arena-table] WARNING: no live backings in registry — \
             all ChunkedKvBacking instances were already dropped at diagnostic capture time. \
             Arena data unavailable."
        );
        return;
    }

    // ── Fixed-width box ──
    // W is the number of characters between the two ║ bookends.
    // Every content line is right-padded to exactly W chars.
    const W: usize = 92;
    let sep = "═".repeat(W);
    let thin = "─".repeat(W);

    // Helper: right-pad a string to W, wrap with ║ bookends, print to stderr.
    let row = |s: &str| {
        if s.len() >= W {
            eprintln!("║{}║", &s[..W]);
        } else {
            eprintln!("║{}{:pad$}║", s, "", pad = W - s.len());
        }
    };

    let backings_with_arenas = tagged_rows
        .iter()
        .map(|(b, _)| *b)
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    eprintln!("\n╔{}╗", sep);
    row(&format!(
        "{:^width$}",
        "ARENA TABLE (all backings)",
        width = W
    ));
    row(&format!(
        "{:^width$}",
        format!(
            "{} live backings, {} dead, {} with arenas",
            live_backings, dead_backings, backings_with_arenas
        ),
        width = W
    ));
    eprintln!("╠{}╣", sep);

    // Column layout:  Bck(4) 2 Idx(4) 2 Type(22) 2 Cap(6) 1 HWM(5) 1 Active(6) 1 Free(5) 1 GPU_MiB(10) 1 Flags(11) = 90 + 2 bookends = 92
    row(&format!(
        " {:>4}  {:>4}  {:<22}  {:>6} {:>5} {:>6} {:>5} {:>10} {:<11}",
        "Bck", "Idx", "Type", "Cap", "HWM", "Active", "Free", "GPU MiB", "Flags"
    ));
    eprintln!("║{}║", thin);

    if tagged_rows.is_empty() {
        row(&format!("{:^width$}", "(no arenas allocated)", width = W));
    } else {
        for (backing_idx, r) in &tagged_rows {
            let mib = format!("{:.1}", r.gpu_bytes as f64 / (1024.0 * 1024.0));
            let flags = if r.is_tombstone {
                "TOMBSTONE"
            } else if r.is_full() {
                "FULL"
            } else if r.is_empty() {
                "empty"
            } else {
                ""
            };
            row(&format!(
                " {:>4}  {:>4}  {:<22}  {:>6} {:>5} {:>6} {:>5} {:>10} {:<11}",
                backing_idx,
                r.arena_idx,
                r.type_label,
                r.capacity,
                r.hwm,
                r.active,
                r.free_list,
                mib,
                flags
            ));
        }
    }

    // ── Summary by type (aggregated across all backings) ──
    eprintln!("╠{}╣", sep);
    row(&format!("{:^width$}", "Summary by type", width = W));
    eprintln!("║{}║", thin);

    let mut by_type: BTreeMap<String, (usize, usize, usize, usize, usize)> = BTreeMap::new();
    let mut tombstone_count = 0usize;
    let mut full_count = 0usize;
    let mut empty_count = 0usize;

    for (_, r) in &tagged_rows {
        if r.is_tombstone {
            tombstone_count += 1;
            continue;
        }
        let e = by_type
            .entry(r.type_label.clone())
            .or_insert((0, 0, 0, 0, 0));
        e.0 += 1; // arena count
        e.1 += r.active;
        e.2 += r.capacity;
        e.3 += r.gpu_bytes;
        e.4 += r.free_list;
        if r.is_full() {
            full_count += 1;
        }
        if r.is_empty() {
            empty_count += 1;
        }
    }

    for (label, (count, active, capacity, bytes, frag)) in &by_type {
        let mib = *bytes as f64 / (1024.0 * 1024.0);
        let frag_pct = if *capacity > 0 {
            (*frag as f64 / *capacity as f64) * 100.0
        } else {
            0.0
        };
        row(&format!(
            "  {:<20} arenas={:<4} used={}/{:<6} free_list={:<4} ({:>4.1}%) {:>8.1} MiB",
            label, count, active, capacity, frag, frag_pct, mib
        ));
    }

    let total_gpu_mib: f64 =
        tagged_rows.iter().map(|(_, r)| r.gpu_bytes).sum::<usize>() as f64 / (1024.0 * 1024.0);
    eprintln!("║{}║", thin);
    row(&format!(
        "  Tombstoned: {:<4}  Full: {:<4}  Empty: {:<4}  Total GPU: {:>10.1} MiB",
        tombstone_count, full_count, empty_count, total_gpu_mib
    ));
    eprintln!("╚{}╝", sep);
}
