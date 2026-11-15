//! `migrate-index-q8`: the one-time rewrite of a store's QSA index keys from
//! `F32` to Q8_0 (`docs/qsa_index_q8.md` §8), run against a stopped daemon's
//! substrate.
//!
//! `--dry-run` opens the store read-only and counts. A run rewrites every
//! `F32` index this node wrote — a later record under a new stamp, which
//! supersedes the one it replaces — commits, then opens the store again and
//! counts what is left: none of this node's own may remain. The superseded
//! records are dead weight maintenance reclaims when the daemon next runs.

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_conversation::persistence::index_q8_migration::{migrate, survey, Report, Tally};

const MIB: f64 = 1024.0 * 1024.0;

fn row(name: &str, t: &Tally) {
    println!(
        "  {name:<28} {:>8} {:>8} {:>11.1} {:>11.1} {:>9} {:>11}",
        t.current,
        t.f32,
        t.f32_bytes as f64 / MIB,
        t.q8_bytes as f64 / MIB,
        t.foreign,
        t.unreadable,
    );
}

fn print(title: &str, report: &Report) {
    println!("{title}");
    println!(
        "  {:<28} {:>8} {:>8} {:>11} {:>11} {:>9} {:>11}",
        "records", "Q8_0", "F32", "F32 MiB", "as Q8 MiB", "foreign", "unreadable"
    );
    row("turn and section index pages", &report.pages);
    row("snapshots and checkpoints", &report.recurrent);
    for (stream, why) in &report.unreadable {
        println!("  unreadable: stream {stream:#018x}: {why}");
    }
    let f = &report.fidelity;
    if f.keys > 0 {
        println!(
            "  keys held as Q8_0: {} — signal to noise {:.1} dB, cosine to the F32 key mean \
             {:.6}, least {:.6}",
            f.keys,
            f.snr_db(),
            f.cos_mean(),
            f.cos_min,
        );
    }
}

/// Count, and unless `dry_run` rewrite, the `F32` indexes of the store under
/// `workspace` (the directory holding `substrate/`).
pub fn migrate_index_q8(workspace: &Path, dry_run: bool) -> Result<()> {
    if dry_run {
        let found = survey(workspace)
            .with_context(|| format!("reading the substrate under {}", workspace.display()))?;
        print("dry run — nothing written:", &found);
        println!("would rewrite: {}", found.to_rewrite());
        return Ok(());
    }
    let did = migrate(workspace)
        .with_context(|| format!("migrating the substrate under {}", workspace.display()))?;
    print("rewritten:", &did);
    let left = survey(workspace).context("reading the store back")?;
    print("the store as it now reads:", &left);
    if left.to_rewrite() != 0 {
        bail!(
            "{} F32 index record(s) of this node's own are still what a reload reads",
            left.to_rewrite()
        );
    }
    let foreign = left.pages.foreign + left.recurrent.foreign;
    if foreign != 0 {
        println!(
            "{foreign} F32 index page(s) were written by another member and are left for it to \
             rewrite; until then this build reads those turns' K/V with no index"
        );
    }
    println!("migrated: {}", did.to_rewrite());
    Ok(())
}
