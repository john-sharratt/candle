//! Dump the projection events for a conversation (timeline) and check which
//! tool section(s) each projection brought into scope — the provenance link.
//!
//! ```text
//! cargo run -p zend --example dump_projections --release -- <workspace> <stream-id | substring>
//! ```

use std::path::PathBuf;

use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::{StreamDecl, StreamId};
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::{decode_events, SystemItem};
use candle_conversation::substrate::Substrate;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let sel = std::env::args().nth(2).unwrap_or_default();

    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate at {}: {e}", workspace.display()))?;
    let tok = Tokenizer::from_file(workspace.join(".substrate").join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // Resolve the target stream: exact id if `sel` parses as u64, else the first
    // turn whose decoded text contains `sel`.
    let by_id: Option<u64> = sel.parse::<u64>().ok();
    let mut target: Option<(u64, u64, u32)> = None; // (stream_id, timeline_id, turn_index)
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(t)) = &e.decl else {
            continue;
        };
        let hit = match by_id {
            Some(id) => sid.0 == id,
            None => {
                let text = persistence
                    .read_tokens(&substrate, sid)
                    .ok()
                    .flatten()
                    .and_then(|b| decode_token_ids(&b).ok())
                    .map(|ids| tok.decode(&ids, false).unwrap_or_default())
                    .unwrap_or_default();
                !sel.is_empty() && text.contains(&sel)
            }
        };
        if hit {
            target = Some((sid.0, t.timeline_id, t.turn_index));
            break;
        }
    }
    let Some((sid0, timeline_id, _)) = target else {
        anyhow::bail!("no turn matched {:?}", sel);
    };
    println!("target stream {sid0}  timeline {timeline_id}\n");

    // Every turn stream in this timeline, ordered by turn_index.
    let mut turns: Vec<(u32, StreamId)> = substrate
        .all_streams()
        .filter_map(|(sid, e)| match &e.decl {
            Some(StreamDecl::Turn(t)) if t.timeline_id == timeline_id => Some((t.turn_index, sid)),
            _ => None,
        })
        .collect();
    turns.sort_by_key(|(ti, _)| *ti);
    println!(
        "conversation has {} turn stream(s) in this timeline",
        turns.len()
    );

    let mut total_events = 0usize;
    for (ti, sid) in &turns {
        let blob = substrate
            .stream_of(*sid)
            .and_then(|s| s.projection_events.clone());
        let Some(blob) = blob else {
            println!(
                "\nturn_index {ti} (stream {})  — NO projection_events record",
                sid.0
            );
            continue;
        };
        let events = decode_events(&blob);
        total_events += events.len();
        println!(
            "\nturn_index {ti} (stream {})  — {} projection event(s)",
            sid.0,
            events.len()
        );
        for (ei, ev) in events.iter().enumerate() {
            // Selected tool/collection sections for this projection.
            let mut selected: Vec<(String, String, u32)> = Vec::new(); // (collection, section, tokens)
            let mut skipped_count = 0usize;
            let mut bare_sections: Vec<String> = Vec::new();
            for item in &ev.selection.system {
                match item {
                    SystemItem::Collection { name, sections } => {
                        for s in sections {
                            if s.selected {
                                selected.push((name.clone(), s.name.clone(), s.tokens));
                            } else {
                                skipped_count += 1;
                            }
                        }
                    }
                    SystemItem::Section { name, .. } => bare_sections.push(name.clone()),
                    SystemItem::Glue { .. } => {}
                }
            }
            let bucket_labels: Vec<String> = ev
                .buckets
                .iter()
                .map(|b| format!("{}={}t", b.label, b.tokens))
                .collect();
            println!(
                "  [{ei}] token {} ({} materialized / {} substrate)  buckets: {}",
                ev.start_token,
                ev.materialized_tokens,
                ev.substrate_tokens,
                bucket_labels.join(", ")
            );
            println!(
                "       collection-selected sections ({} selected, {} skipped):",
                selected.len(),
                skipped_count
            );
            for (coll, name, tk) in &selected {
                println!("         ✓ [{coll}] {name}  ({tk}t)");
            }
            if !bare_sections.is_empty() {
                println!("       bare system sections: {}", bare_sections.join(", "));
            }
            println!(
                "       selected turns: {}",
                ev.selection
                    .turns
                    .iter()
                    .map(|t| format!("{}/{}#{}", t.layer, t.group, t.index))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    println!("\ntotal projection events across the conversation: {total_events}");
    Ok(())
}
