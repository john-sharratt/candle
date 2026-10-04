//! The whole-card VRAM decomposition, read off the span's own accounting.
//!
//! Printed twice per gate: at the end of each config's decode phase, while every
//! session of the row is still alive and KV sits at its high-water for that row,
//! and once at the end of the run. The decode-end snapshot is the one that says
//! where the card went while decode was running — the end-of-run one sees the
//! ground the last row's sessions have already given back.
//!
//! Read off the reservation rather than reconstructed from slot and region
//! counts: the two round in different units, and a boundary that moved eighty
//! times accumulates exactly the sort of gap that otherwise gets inferred away.

use candle::Device;
use candle_nn::kv_cache::{region_stats, REGION_BYTES};

use crate::models::expert_lre::PipelineStats;

fn mib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

/// Print device 0's span decomposition under `title`. `experts` adds the expert
/// cache's resident bytes inside the weight zone and the hit rate that residency
/// bought; `None` for a model without one.
pub fn print_span(title: &str, experts: Option<&PipelineStats>) {
    let Some(rs) = region_stats(0) else {
        return;
    };
    println!("\n=== Span (device 0): {title} ===");
    let card = Device::new_cuda(0).and_then(|d| d.mem_get_info()).ok();
    if let Some((free, total)) = card {
        println!(
            "  CARD                {:>9.1} MiB total | {:>7.1} free | {:>7.1} in use",
            mib(total),
            mib(free),
            mib(total - free),
        );
    }
    println!("  reserved            {:>9.1} MiB", mib(rs.reserved_bytes));
    let kv = rs.total * REGION_BYTES;
    // Whatever the reservation holds that is neither side of the moving
    // boundary: the dense weights, loaded before the boundary existed.
    let dense = rs
        .reserved_bytes
        .saturating_sub(rs.weight_bytes)
        .saturating_sub(kv)
        .saturating_sub(rs.slack_bytes);
    println!(
        "    dense block       {:>9.1} MiB   (loaded before the boundary, immovable)",
        mib(dense)
    );
    println!(
        "    KV regions        {:>9.1} MiB   ({} x {:.0} MiB: live {} of which span-tenant {}, \
         free {}, blocked {}, frontier {})",
        mib(kv),
        rs.total,
        mib(REGION_BYTES),
        rs.live,
        rs.span_tenant,
        rs.free,
        rs.blocked,
        rs.live_watermark,
    );
    println!(
        "      live KV         {:>9.1} MiB   (arenas {:.1} + span tenants {:.1})",
        mib(rs.live * REGION_BYTES),
        mib(rs.live.saturating_sub(rs.span_tenant) * REGION_BYTES),
        mib(rs.span_tenant * REGION_BYTES),
    );
    if rs.transient_bytes > 0 {
        println!(
            "    wave tier         {:>9.1} MiB   (placed now, ceiling {} regions)",
            mib(rs.transient_bytes),
            rs.transient_ceiling,
        );
    }
    println!(
        "    weight zone       {:>9.1} MiB   (the expert side of the boundary)",
        mib(rs.weight_bytes)
    );
    if let Some(s) = experts {
        println!(
            "      experts resident {:>8.1} MiB   ({} slots of {:.2} MiB; zone {:.1}, min {:.1}, \
             max {:.1}) | hit {:.1}% | warm tier {} of {} experts | misses pinned {} cold {}",
            mib(s.resident_vram_bytes),
            s.resident_vram_bytes
                .checked_div(s.expert_slot_bytes)
                .unwrap_or(0),
            mib(s.expert_slot_bytes),
            mib(s.zone_bytes),
            mib(s.zone_min_bytes),
            mib(s.zone_max_bytes),
            s.hit_rate(),
            s.warm_slots,
            s.total_experts,
            s.worker_pinned,
            s.worker_cold,
        );
    }
    println!(
        "    unusable slack    {:>9.1} MiB   (span tail the region count rounds off)",
        mib(rs.slack_bytes)
    );
    if let Some((free, total)) = card {
        let in_use = total - free;
        let outside = in_use.saturating_sub(rs.reserved_bytes);
        match rs.pre_reservation_in_use_bytes {
            Some(base) => {
                let base = base as usize;
                println!(
                    "  outside the span    {:>9.1} MiB   (baseline before reservation {:.1}, \
                     since {:.1})",
                    mib(outside),
                    mib(base),
                    mib(outside.saturating_sub(base)),
                );
            }
            None => println!(
                "  outside the span    {:>9.1} MiB   (no pre-reservation baseline recorded)",
                mib(outside)
            ),
        }
    }
    println!(
        "  peak KV live        {:>9.1} MiB   ({} regions)   granule {:.0} MiB",
        mib(rs.peak_live * REGION_BYTES),
        rs.peak_live,
        mib(rs.granularity),
    );
}
