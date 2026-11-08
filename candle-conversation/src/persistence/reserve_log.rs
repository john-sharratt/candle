//! The `kv_reserve` line a persistence pass writes
//! (`docs/kv_drain_under_pressure.md` §2b): the native backlog the reserve is
//! sized from, every class's target, free slots, owed arenas and factor, and
//! the reserve against the KV ground and its cap — so the factor is a number in
//! the log, and the drain rate is the difference between two lines.

use candle_nn::kv_cache::ReserveReport;

const MIB: usize = 1 << 20;

/// One class per field, `label:target/free+owed@factor`, a `!` for a class
/// held at its cap.
pub(super) fn reserve_classes(report: &ReserveReport) -> String {
    report
        .classes
        .iter()
        .map(|c| {
            format!(
                "{}:{}/{}+{}@{:.3}{}",
                c.key.label(),
                c.target_slots,
                c.free_slots,
                c.owed_arenas,
                c.factor,
                if c.capped { "!" } else { "" },
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Write the pass's `kv_reserve` line, with what the pass moved hot→warm and
/// how long it took.
pub(super) fn log_reserve(report: &ReserveReport, moved: usize, moved_bytes: u64, pass_ms: u64) {
    tracing::info!(
        target: "candle_conversation::kv_reserve",
        pending_slots = report.pending_slots,
        pending_mib = report.pending_bytes / MIB,
        reserve_free_mib = report.free_bytes() / MIB,
        cap_mib = report.cap_bytes / MIB,
        kv_mib = report.kv_bytes / MIB,
        moved_residences = moved,
        moved_mib = moved_bytes / MIB as u64,
        pass_ms,
        classes = %reserve_classes(report),
        "kv_reserve: pass",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::kv_cache::{ArenaKey, ArenaLocation, ReserveClass};

    /// **Each class reads as `label:target/free+owed@factor`**, in the order
    /// the report holds them, capped ones marked.
    #[test]
    fn each_class_reads_as_target_free_owed_and_factor() {
        let band = ArenaKey::for_record_stride(ArenaLocation::Gpu, 1536);
        let report = ReserveReport {
            classes: vec![
                ReserveClass {
                    key: band,
                    target_slots: 12_000,
                    free_slots: 9_000,
                    owed_arenas: 1,
                    factor: 0.25,
                    capped: false,
                },
                ReserveClass {
                    key: band,
                    target_slots: 40,
                    free_slots: 40,
                    owed_arenas: 0,
                    factor: 1.0,
                    capped: true,
                },
            ],
            ..ReserveReport::default()
        };
        assert_eq!(
            reserve_classes(&report),
            "record1536:12000/9000+1@0.250 record1536:40/40+0@1.000!"
        );
    }
}
