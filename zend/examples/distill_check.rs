//! Verify the archive → distill storage path over a real substrate.
//!
//! For every distilled timeline, reports its `DistillMode` and — from the
//! reloaded stream index — whether its turns actually shed to that mode:
//!   TextOnly       → tokens kept, sig + KV chunks + projections dropped.
//!   ProvenanceOnly → sig kept, tokens + KV chunks dropped.
//!
//! Run with the daemon STOPPED (it opens the substrate for recovery):
//! ```text
//! cargo run -p zend --example distill_check --release -- [workspace]
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use candle_conversation::persistence::record::DistillMode;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;

#[derive(Default, Clone)]
struct Acc {
    turns: usize,
    tokens: usize,
    sig: usize,
    proj: usize,
    chunks: usize,
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut substrate = Substrate::new();
    let _p = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate: {e}"))?;

    let distilled: BTreeMap<u64, DistillMode> = substrate
        .distilled_timelines()
        .iter()
        .map(|(t, m)| (t.raw(), *m))
        .collect();

    // Per-timeline turn accounting from the reloaded (post-compaction) index.
    let mut per_tl: BTreeMap<u64, Acc> = BTreeMap::new();
    for (_sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(d)) = &e.decl else {
            continue;
        };
        let a = per_tl.entry(d.timeline_id).or_default();
        a.turns += 1;
        a.tokens += e.tokens.is_some() as usize;
        a.sig += e.wide_q_sigs.is_some() as usize;
        a.proj += e.projection_events.is_some() as usize;
        a.chunks += e.chunks.len();
    }

    let convs: BTreeMap<u64, (String, bool)> = substrate
        .known_conversations()
        .into_iter()
        .map(|(tl, _cid, label, archived, _order)| (tl.raw(), (label, archived)))
        .collect();

    println!("=== {} distilled timelines ===\n", distilled.len());
    let mut ok = 0usize;
    for (tl, mode) in &distilled {
        let a = per_tl.get(tl).cloned().unwrap_or_default();
        let (label, archived) = convs.get(tl).cloned().unwrap_or_default();
        // Expected shed per mode.
        let shed_ok = match mode {
            DistillMode::TextOnly => {
                a.tokens == a.turns && a.sig == 0 && a.chunks == 0 && a.proj == 0
            }
            DistillMode::ProvenanceOnly => a.sig == a.turns && a.tokens == 0 && a.chunks == 0,
        };
        ok += shed_ok as usize;
        println!(
            "{} tl={tl}  {mode:?}  archived={archived}  \"{}\"",
            if shed_ok { "OK  " } else { "BAD " },
            if label.is_empty() {
                "(untitled)"
            } else {
                &label
            }
        );
        println!(
            "     turns={} tokens={} sig={} proj={} kv_chunks={}",
            a.turns, a.tokens, a.sig, a.proj, a.chunks
        );
    }
    println!(
        "\n{ok}/{} distilled timelines shed exactly as their mode dictates",
        distilled.len()
    );
    Ok(())
}
