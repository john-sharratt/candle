//! `substrate_inspect completeness` — which turns hold every record they
//! declare, by the member that wrote them.
//!
//! A turn is complete when it holds its `Tokens` and every `Chunk` of its block
//! range on every K/V layer — the test a member applies to another member's
//! turn before adopting it (`persistence/remote/completeness.rs`). A member
//! never applies it to its own turns, so a turn whose K/V its own writer never
//! put in the log reads as whole there and as a permanent gap everywhere else.
//! Grouped by writer, this shows which side a gap comes from: a writer's own
//! turns short of chunks on its own disk were never written, and there is
//! nothing for any other member to fetch.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_conversation::persistence::remote::turn_is_complete;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::TimelineId;
use candle_conversation::substrate::Substrate;

/// One writer's turns, counted.
#[derive(Default)]
struct Tally {
    turns: usize,
    complete: usize,
    /// Declared no K/V (a summary node, a placeholder) or distilled.
    declared_empty: usize,
    no_tokens: usize,
    /// Short of chunks, by how many blocks' worth they hold of the declared
    /// count, in tenths: `0` holds none, `9` most.
    short: BTreeMap<usize, usize>,
    /// Of the short ones, on a tombstoned timeline.
    short_tombstoned: usize,
    /// A few short turns, for a closer look: `(timeline, turn, held, wanted)`.
    samples: Vec<(u64, u32, usize, usize)>,
    /// By projection layer id: `(turns, complete, holding no chunk at all)`.
    by_layer: BTreeMap<u32, (usize, usize, usize)>,
}

/// Count every turn of the segmented store at `substrate_dir`, against
/// `kv_layers` K/V layers.
pub fn completeness(substrate_dir: &Path, kv_layers: usize) -> Result<()> {
    let workspace = substrate_dir
        .parent()
        .context("the substrate directory has no parent to open it from")?;
    let mut substrate = Substrate::new();
    let persistence =
        SubstratePersistence::open_in_with_substrate_read_only(workspace, &mut substrate)
            .with_context(|| format!("opening {} read-only", substrate_dir.display()))?;
    let local = persistence.minter().node();

    // Every chunk the stream index holds, asked of the listing a catch-up
    // answer is built from (§7.2): one it reports gone is one an answer tells
    // every other member is gone, and none of them asks for it again.
    let mut listed = 0usize;
    let mut listed_gone: BTreeMap<u64, usize> = BTreeMap::new();
    let mut gone_samples: Vec<(u64, u64, u64)> = Vec::new();
    for (key, stream) in substrate.all_physical_streams() {
        for (&idx, id) in stream.ids.chunks.iter() {
            if id.is_none() {
                continue;
            }
            if persistence.list(id.origin, id.oseq, id.oseq).records.is_empty() {
                *listed_gone.entry(id.origin).or_default() += 1;
                if gone_samples.len() < 8 {
                    gone_samples.push((key.stream_id.0, idx, id.oseq));
                }
            } else {
                listed += 1;
            }
        }
    }
    println!("\nheld chunks as a catch-up answer lists them: {listed} listed");
    for (origin, n) in &listed_gone {
        println!("  {n} of origin {origin:016x} held but listed GONE");
    }
    for (stream, idx, oseq) in &gone_samples {
        println!("  e.g. stream {stream:#018x} chunk {idx}: oseq {oseq}");
    }

    let mut by_writer: BTreeMap<u64, Tally> = BTreeMap::new();
    for (_, stream) in substrate.all_streams() {
        let Some(StreamDecl::Turn(decl)) = stream.decl.as_ref() else {
            continue;
        };
        let Some(timeline) = TimelineId::from_raw(decl.timeline_id) else {
            continue;
        };
        let writer = substrate.turn_mark(timeline, decl.turn_index).writer;
        let tally = by_writer.entry(writer).or_default();
        tally.turns += 1;
        let blocks = decl.block_end.saturating_sub(decl.block_start) as usize;
        let distilled = substrate.distill_mode(timeline).is_some();
        let complete = turn_is_complete(decl, stream, kv_layers, distilled);
        let layer = tally.by_layer.entry(decl.layer_id).or_default();
        layer.0 += 1;
        if complete {
            layer.1 += 1;
        } else if stream.chunks.is_empty() {
            layer.2 += 1;
        }
        if complete {
            if distilled || blocks == 0 {
                tally.declared_empty += 1;
            } else {
                tally.complete += 1;
            }
            continue;
        }
        if stream.tokens.is_none() {
            tally.no_tokens += 1;
        }
        let wanted = kv_layers * blocks;
        let held = stream.chunks.len();
        if held != wanted {
            *tally.short.entry(held * 10 / wanted.max(1)).or_default() += 1;
            if substrate.is_tombstoned(timeline) {
                tally.short_tombstoned += 1;
            }
            if tally.samples.len() < 8 {
                tally
                    .samples
                    .push((decl.timeline_id, decl.turn_index, held, wanted));
            }
        }
    }

    println!("kv layers: {kv_layers}   this store's node: {local:016x}");
    // What this store says it holds of every origin — where a catch-up asks
    // from (`docs/replicated_substrate.md` §7.2). A record counted here is
    // never asked for again, whether or not its bytes are in the log.
    println!("\napplied, by origin:");
    for (origin, applied) in persistence.applied_set().origins() {
        let gaps = applied.gaps();
        let missing: u64 = gaps.iter().map(|(s, e)| e - s + 1).sum();
        println!(
            "  {origin:016x}: through {} top {} — {} gaps, {missing} oseqs uncovered",
            applied.through(),
            applied.top(),
            gaps.len()
        );
    }
    for (writer, t) in &by_writer {
        let who = match *writer {
            0 => "unmarked".to_string(),
            w if w == local => format!("{w:016x} (this node)"),
            w => format!("{w:016x}"),
        };
        let short: usize = t.short.values().sum();
        println!(
            "\nwriter {who}: {} turns — {} complete, {} declare no K/V, {} short of chunks \
             ({} on tombstoned timelines), {} without tokens",
            t.turns, t.complete, t.declared_empty, short, t.short_tombstoned, t.no_tokens
        );
        for (tenths, n) in &t.short {
            println!("  holds {:>3}%..: {n}", tenths * 10);
        }
        for (layer, (turns, complete, none)) in &t.by_layer {
            println!("  layer {layer}: {turns} turns, {complete} complete, {none} hold no chunk");
        }
        for (tl, turn, held, wanted) in &t.samples {
            println!("  e.g. timeline {tl:#018x} turn {turn}: {held} of {wanted} chunks");
        }
    }
    Ok(())
}
