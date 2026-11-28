//! `substrate_inspect compact-check` — whether a compaction keeps what the
//! substrate locates, run on a **copy** of a store.
//!
//! Opens the store writable, notes every stream the substrate holds — its
//! declaration's identity and the chunks it locates — compacts it, and notes
//! them again. Every stream the re-apply lost, or whose chunks it stopped
//! locating, is printed with the identities of its declaration and its chunks,
//! which is what decides whether the re-apply takes a record back.
//!
//! It rewrites the store it is given: point it at a copy, never at a live one.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_conversation::persistence::identity::RecordIdentity;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::{StreamKey, Substrate};

/// What the substrate held of one stream.
struct Held {
    declared: bool,
    decl: RecordIdentity,
    chunks: usize,
    chunk_ids: Vec<RecordIdentity>,
}

fn held(substrate: &Substrate) -> BTreeMap<StreamKey, Held> {
    substrate
        .all_physical_streams()
        .map(|(key, rt)| {
            (
                key,
                Held {
                    declared: rt.decl.is_some(),
                    decl: rt.ids.decl,
                    chunks: rt.chunks.len(),
                    chunk_ids: rt.ids.chunks.values().copied().collect(),
                },
            )
        })
        .collect()
}

fn show(id: &RecordIdentity) -> String {
    format!(
        "origin={:016x} oseq={} gen={} epoch={} hlc={}",
        id.origin, id.oseq, id.gen, id.epoch, id.hlc
    )
}

/// Compact the store at `substrate_dir` — a copy — and report what the
/// substrate stopped holding.
pub fn compact_check(substrate_dir: &Path) -> Result<()> {
    let workspace = substrate_dir
        .parent()
        .context("the substrate directory has no parent to open it from")?;
    let mut substrate = Substrate::new();
    let mut persistence =
        SubstratePersistence::open_in_with_substrate(workspace, &mut substrate)
            .with_context(|| format!("opening {} writable", substrate_dir.display()))?;
    let before = held(&substrate);
    let chunks_before: usize = before.values().map(|h| h.chunks).sum();
    println!(
        "before: {} streams, {} declared, {} chunks located, node {:016x}",
        before.len(),
        before.values().filter(|h| h.declared).count(),
        chunks_before,
        persistence.node()
    );

    persistence.compact(&mut substrate, None)?;
    let after = held(&substrate);
    let chunks_after: usize = after.values().map(|h| h.chunks).sum();
    println!(
        "after:  {} streams, {} declared, {} chunks located",
        after.len(),
        after.values().filter(|h| h.declared).count(),
        chunks_after
    );

    let mut lost = 0usize;
    for (key, b) in &before {
        let a = after.get(key);
        let a_chunks = a.map_or(0, |a| a.chunks);
        let a_declared = a.is_some_and(|a| a.declared);
        if a_chunks == b.chunks && a_declared == b.declared {
            continue;
        }
        lost += 1;
        if lost <= 20 {
            println!(
                "stream {:016x} writer {:016x}: declared {} -> {}, chunks {} -> {}",
                key.stream_id.0, key.writer, b.declared, a_declared, b.chunks, a_chunks
            );
            println!("  decl  {}", show(&b.decl));
            for id in b.chunk_ids.iter().take(3) {
                println!("  chunk {}", show(id));
            }
            if let Some(a) = a {
                println!("  after decl {}", show(&a.decl));
            }
        }
    }
    println!("{lost} stream(s) changed by the compaction's re-apply");
    Ok(())
}
