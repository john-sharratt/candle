//! `substrate_inspect liveness` — the maintenance liveness audit, printed.
//!
//! Opens the store read-only, builds the same in-RAM index a reload does, and
//! runs [`SubstratePersistence::liveness_audit`]: every record of every segment
//! classified as counted live or not, carried by maintenance or not. A record
//! carried but counted dead is rewritten by every op and makes its new segment
//! look reclaimable — the churn this exists to find; a record counted live but
//! not carried is one a drop would lose.

use std::path::Path;

use anyhow::{Context, Result};
use candle_conversation::persistence::liveness_audit::AuditCell;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;

const MIB: f64 = 1024.0 * 1024.0;

fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

/// Audit the segmented store at `substrate_dir` (the `substrate` directory).
pub fn liveness(substrate_dir: &Path) -> Result<()> {
    let workspace = substrate_dir
        .parent()
        .context("the substrate directory has no parent to open it from")?;
    let mut substrate = Substrate::new();
    let persistence =
        SubstratePersistence::open_in_with_substrate_read_only(workspace, &mut substrate)
            .with_context(|| format!("opening {} read-only", substrate_dir.display()))?;
    let audit = persistence.liveness_audit(&substrate)?;

    println!("by record type (MiB):");
    println!(
        "  {:<18} {:>9} {:>10} {:>10} {:>10}   {:>22}   {:>22}",
        "type",
        "records",
        "on disk",
        "counted",
        "carried",
        "carried, counted dead",
        "counted, not carried"
    );
    for (rt, c) in audit.by_type() {
        println!(
            "  {:<18} {:>9} {:>10.1} {:>10.1} {:>10.1}   {:>8} rec {:>9.1}   {:>8} rec {:>9.1}",
            format!("{rt:?}"),
            c.records,
            mib(c.bytes),
            mib(c.counted),
            mib(c.carried),
            c.carried_uncounted_records,
            mib(c.carried_uncounted),
            c.counted_uncarried_records,
            mib(c.counted_uncarried),
        );
    }

    println!("\nby segment — dead as maintenance counts it, against dead as it carries:");
    println!(
        "  {:>8} {:>10} {:>14} {:>12} {:>16} {:>16}",
        "segment", "MiB", "counted dead%", "true dead%", "churn MiB", "unsafe MiB"
    );
    let mut total = AuditCell::default();
    for (seg, c) in audit.by_segment() {
        println!(
            "  {:>8} {:>10.1} {:>13.1}% {:>11.1}% {:>16.1} {:>16.1}",
            seg.0,
            mib(c.bytes),
            pct(c.bytes - c.counted.min(c.bytes), c.bytes),
            pct(c.bytes - c.carried.min(c.bytes), c.bytes),
            mib(c.carried_uncounted),
            mib(c.counted_uncarried),
        );
        total.bytes += c.bytes;
        total.counted += c.counted;
        total.carried += c.carried;
        total.carried_uncounted += c.carried_uncounted;
        total.counted_uncarried += c.counted_uncarried;
    }
    println!(
        "  {:>8} {:>10.1} {:>13.1}% {:>11.1}% {:>16.1} {:>16.1}",
        "all",
        mib(total.bytes),
        pct(total.bytes - total.counted.min(total.bytes), total.bytes),
        pct(total.bytes - total.carried.min(total.bytes), total.bytes),
        mib(total.carried_uncounted),
        mib(total.counted_uncarried),
    );
    if total.carried_uncounted == 0 && total.counted_uncarried == 0 {
        println!("\nCONSISTENT — the count and the carry agree on every record");
    } else {
        println!(
            "\nMISMATCH — {:.1} MiB carried but counted dead (churn), {:.1} MiB counted live but not carried (lost on drop)",
            mib(total.carried_uncounted),
            mib(total.counted_uncarried)
        );
    }
    Ok(())
}
