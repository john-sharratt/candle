//! Where the card's memory is, at a named moment of a gate run.
//!
//! One figure per tenant of the device, read off the span's own accounting
//! rather than reconstructed from slot and region counts — the two round in
//! different units, and a gap between them is exactly what gets inferred away.
//! Taken after each config's prefill and after its decode, so a quant's
//! distribution can be compared across the two phases and against the other
//! quants: how far the expert zone grew or was squeezed, how much KV went live,
//! and what the rest of the card holds outside the reservation.

use candle::{Device, Result};

const MIB: f64 = 1024.0 * 1024.0;

/// The device's memory, split by tenant, at one moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanSnapshot {
    /// The whole card, as the driver reports it.
    pub card_total: usize,
    pub card_free: usize,
    /// The span reservation: dense block + KV regions + expert zone + slack.
    pub reserved: usize,
    /// The expert side of the moving boundary.
    pub weight: usize,
    /// Dense weights, loaded before the boundary existed and immovable.
    pub dense: usize,
    /// KV region ground, and how it is split.
    pub region_bytes: usize,
    pub kv_regions: usize,
    pub kv_live: usize,
    pub kv_free: usize,
    pub kv_blocked: usize,
    /// The span tail the region count rounds off.
    pub slack: usize,
    /// What the CUDA stream-ordered pool has reserved from the driver, and
    /// how much of that is live. The span is VMM-mapped and not in the pool,
    /// so all of it is outside the span; `reserved - used` is memory the pool
    /// holds cached for reuse and the rest of the card cannot have.
    pub pool_reserved: usize,
    pub pool_used: usize,
    /// Free VRAM when the device was created, before any weight: the card less
    /// the CUDA context and other processes. `None` when not recorded.
    pub init_free: Option<usize>,
}

impl SpanSnapshot {
    /// The split on `device` now, or `None` where there is no span (no CUDA,
    /// or no model has placed one).
    #[cfg(not(feature = "cuda"))]
    pub fn capture(_device: &Device) -> Result<Option<Self>> {
        Ok(None)
    }

    /// See the non-CUDA form.
    #[cfg(feature = "cuda")]
    pub fn capture(device: &Device) -> Result<Option<Self>> {
        let Device::Cuda(_) = device else {
            return Ok(None);
        };
        let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
            return Ok(None);
        };
        let Some(rs) = candle_nn::kv_cache::region_stats(gpu_id) else {
            return Ok(None);
        };
        let (card_free, card_total) = device.mem_get_info()?;
        let Device::Cuda(cuda) = device else {
            unreachable!("matched above")
        };
        let pool_reserved = cuda.pool_reserved_bytes()?;
        let pool_used = cuda.pool_used_bytes()?;
        let init_free = candle::gpu_memory::device_init_free(gpu_id);
        let region_bytes = candle_nn::kv_cache::REGION_BYTES;
        let kv = rs.total * region_bytes;
        Ok(Some(Self {
            card_total,
            card_free,
            reserved: rs.reserved_bytes,
            weight: rs.weight_bytes,
            dense: rs
                .reserved_bytes
                .saturating_sub(rs.weight_bytes)
                .saturating_sub(kv)
                .saturating_sub(rs.slack_bytes),
            region_bytes,
            kv_regions: rs.total,
            kv_live: rs.live,
            kv_free: rs.free,
            kv_blocked: rs.blocked,
            slack: rs.slack_bytes,
            pool_reserved,
            pool_used,
            init_free,
        }))
    }

    /// Card memory in use outside the reservation: the CUDA context, the
    /// allocator pool and anything else the driver holds for the process.
    pub fn outside_span(&self) -> usize {
        (self.card_total - self.card_free).saturating_sub(self.reserved)
    }

    /// The part of [`Self::outside_span`] held before any weight loaded: the
    /// CUDA context and other processes. Zero when not recorded.
    pub fn context_and_others(&self) -> usize {
        self.init_free
            .map_or(0, |f| self.card_total.saturating_sub(f))
    }

    /// Outside the span, in neither the pool nor the context: direct driver
    /// allocations — library workspaces, device tables allocated outside the
    /// pool, and anything loaded straight to the device.
    pub fn outside_other(&self) -> usize {
        self.outside_span()
            .saturating_sub(self.pool_reserved)
            .saturating_sub(self.context_and_others())
    }

    /// The rows of [`print_table`], each a label and this snapshot's value.
    fn rows(&self) -> [(&'static str, String); 13] {
        let mib = |b: usize| format!("{:.0}", b as f64 / MIB);
        let kv = |n: usize| format!("{} ({n})", mib(n * self.region_bytes));
        [
            ("card in use", mib(self.card_total - self.card_free)),
            ("card free", mib(self.card_free)),
            ("outside span", mib(self.outside_span())),
            ("  context + others", mib(self.context_and_others())),
            ("  pool live", mib(self.pool_used)),
            ("  pool cached", mib(self.pool_reserved - self.pool_used)),
            ("  other direct", mib(self.outside_other())),
            ("dense block", mib(self.dense)),
            ("expert zone", mib(self.weight)),
            ("KV live (regions)", kv(self.kv_live)),
            ("KV free (regions)", kv(self.kv_free)),
            ("KV blocked (regions)", kv(self.kv_blocked)),
            ("span slack", mib(self.slack)),
        ]
    }
}

/// Print one column per `(label, snapshot)`, one row per tenant, in MiB.
///
/// `columns` are typically each config's after-prefill and after-decode
/// snapshots side by side; a column whose snapshot is absent prints `-`.
pub fn print_table(title: &str, columns: &[(String, Option<SpanSnapshot>)]) {
    if columns.iter().all(|(_, s)| s.is_none()) {
        return;
    }
    const LABEL_W: usize = 22;
    let width = columns
        .iter()
        .map(|(l, _)| l.len())
        .max()
        .unwrap_or(0)
        .max(12);
    println!("\n=== {title} (MiB) ===");
    let mut header = format!("{:<LABEL_W$}", "");
    for (label, _) in columns {
        header.push_str(&format!(" │ {label:>width$}"));
    }
    println!("{header}");
    let rows: Vec<Option<[(&str, String); 13]>> =
        columns.iter().map(|(_, s)| s.map(|s| s.rows())).collect();
    let names = rows
        .iter()
        .flatten()
        .next()
        .map(|r| r.each_ref().map(|(n, _)| *n))
        .expect("one column has a snapshot");
    for (i, name) in names.iter().enumerate() {
        let mut line = format!("{name:<LABEL_W$}");
        for r in &rows {
            let v = r.as_ref().map_or("-".to_string(), |r| r[i].1.clone());
            line.push_str(&format!(" │ {v:>width$}"));
        }
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> SpanSnapshot {
        SpanSnapshot {
            card_total: 16_000 << 20,
            card_free: 1_000 << 20,
            reserved: 9_000 << 20,
            weight: 3_000 << 20,
            dense: 3_696 << 20,
            region_bytes: 16 << 20,
            kv_regions: 144,
            kv_live: 100,
            kv_free: 30,
            kv_blocked: 14,
            slack: 8 << 20,
            pool_reserved: 3_000 << 20,
            pool_used: 1_200 << 20,
            init_free: Some(15_400 << 20),
        }
    }

    #[test]
    fn outside_span_is_in_use_less_the_reservation() {
        // 15,000 MiB in use against a 9,000 MiB reservation.
        assert_eq!(snap().outside_span(), 6_000 << 20);
    }

    #[test]
    fn outside_span_splits_into_context_pool_and_direct() {
        let s = snap();
        // 16,000 total against 15,400 free at creation.
        assert_eq!(s.context_and_others(), 600 << 20);
        // 6,000 outside, less the 3,000 pool and the 600 context.
        assert_eq!(s.outside_other(), 2_400 << 20);
    }

    #[test]
    fn rows_report_mib_and_region_counts() {
        let r = snap().rows();
        assert_eq!(r[0], ("card in use", "15000".to_string()));
        assert_eq!(r[2], ("outside span", "6000".to_string()));
        assert_eq!(r[5], ("  pool cached", "1800".to_string()));
        assert_eq!(r[8], ("expert zone", "3000".to_string()));
        assert_eq!(r[9], ("KV live (regions)", "1600 (100)".to_string()));
    }
}
