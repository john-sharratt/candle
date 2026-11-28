//! `substrate_inspect cut-after` — cut a stopped store's log back to before a
//! run, through `persistence::tail_cut`.
//!
//! Prints the plan — the segment truncated and where, the segments set aside,
//! the records cut by type, and the latest kept record of the origin by its
//! clock — and applies it only with `--apply`. Plan first, read the figures,
//! then apply: the gap between the latest kept record and the cut is the
//! pause between the two runs, and a cut that lands inside a run shows almost
//! none.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_conversation::persistence::tail_cut::{apply, plan};

/// Plan the cut of the store at `substrate_dir` at `origin`'s (hex) first
/// record at or after `from_ms`, and apply it when `apply_it`.
pub fn cut_after(substrate_dir: &Path, origin: &str, from_ms: u64, apply_it: bool) -> Result<()> {
    let origin = u64::from_str_radix(origin.trim_start_matches("0x"), 16)
        .with_context(|| format!("origin {origin:?} is not a hex node id"))?;
    let Some(cut) = plan(substrate_dir, origin, from_ms)? else {
        println!("nothing of origin {origin:016x} at or after {from_ms}: no cut");
        return Ok(());
    };
    match &cut.truncate {
        Some((id, path, offset)) => println!(
            "truncate segment {id} ({}) at offset {offset}",
            path.display()
        ),
        None => println!("no segment truncated: the cut opens a segment"),
    }
    for (id, path) in &cut.set_aside {
        println!("set aside segment {id} ({})", path.display());
    }
    println!("{} records cut:", cut.records_cut());
    for (ty, n) in &cut.cut {
        println!("  {ty}: {n}");
    }
    let mut by_prefix: BTreeMap<&str, usize> = BTreeMap::new();
    for key in &cut.carried_objects {
        let prefix = key.split('/').next().unwrap_or(key);
        *by_prefix.entry(prefix).or_insert(0) += 1;
    }
    println!(
        "{} custom objects cut that carry a clock from before the run (re-emitted copies):",
        cut.carried_objects.len()
    );
    for (prefix, n) in &by_prefix {
        println!("  {prefix}/…: {n}");
    }
    for key in cut.carried_objects.iter().take(15) {
        println!("  e.g. {key}");
    }
    println!(
        "latest kept record of {origin:016x}: {} ms ({} ms before the cut)",
        cut.last_kept_ms,
        from_ms.saturating_sub(cut.last_kept_ms)
    );
    if apply_it {
        apply(&cut)?;
        println!("applied");
    } else {
        println!("plan only: pass --apply to cut");
    }
    Ok(())
}
